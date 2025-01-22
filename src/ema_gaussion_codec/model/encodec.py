import logging
from dataclasses import dataclass, field

import torch.distributed

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import math
import random
import typing as tp
from dataclasses import dataclass, field
from typing import Tuple,Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from .. import modules as m
from ..criterion.audio_to_mel import Audio2Mel
from ..modules import quantization as qt
from ..modules.discriminator import MultiScaleSTFTDiscriminator,MultiPeriodDiscriminator,MultiScaleDiscriminator,MultiResSpecDiscriminator
from ..task.encodec_train import EncodecTrainConfig, EncodecTrainTask
from .utils import _linear_overlap_add

logger = logging.getLogger(__name__)
EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


@dataclass
class EncodecConfig(FairseqDataclass):
    target_bandwidths: Tuple[float, ...] = field(default=(1.5, 3., 6., 12., 24.), metadata={"help": "target bandwidths"})
    sample_rate: int = field(default=24_000, metadata={"help": "sample rate"})
    channels: int = field(default=1, metadata={"help": "channels"})
    model_norm: str = field(default="weight_norm", metadata={"help": "model norm method"})
    audio_normalize: bool = field(default=False, metadata={"help": "audio normalize"})
    overlap: float = field(default=0.01, metadata={"help": "overlap"})
    casual: bool = field(default=True, metadata={"help": "casual"})
    encoder_ratios: Tuple[int, ...] = field(default=(8, 5, 4, 2), metadata={"help": "ratios"})
    decoder_ratios: Tuple[int, ...] = field(default=(8, 5, 4, 2), metadata={"help": "ratios"})
    encoder_dimension: int = field(default=128, metadata={"help": "encoder dimension"})
    encoder_n_residual_layers: int = field(default=1, metadata={"help": "number of residual layers"})
    decoder_dimension: int = field(default=128, metadata={"help": "decoder dimension"})
    decoder_n_residual_layers: int = field(default=1, metadata={"help": "number of residual layers"})
    final_activation: Optional[str] = field(default=None, metadata={"help": "decoder final activation"})
    encoder_n_filters: int = field(default=32, metadata={"help": "number of filters"})
    decoder_n_filters: int = field(default=32, metadata={"help": "number of filters"})
    random_select: bool = field(default=False, metadata={"help": "random select"})
    full_bandwidth: bool = field(default=False, metadata={"help": "full bandwidth"})
    
    # codebook
    codebook_dim: int = field(default=256, metadata={"help": "codebook dimension(mean and logvar), if codebook_dim is not None, encoder dim will proj to the codebook_dim"})
    bins: int = field(default=1024, metadata={"help": "bins"})
    n_q: int = field(default=0, metadata={"help": "number of codebook"})
    lstm: int = field(default=2, metadata={"help": "lstm layer number"})
    transformer: int = field(default=0, metadata={"help": "transformer layer number"})
    n_head: int = field(default=8, metadata={"help": "multi head attention"})
    activation: str = field(default="ELU", metadata={"help": "activation function"})
    activation_params: tp.Dict[str, float] = field(default_factory=lambda: {"alpha": 1.0}, metadata={"help": "activation params"})
   
    # discriminator
    discriminator: str = field(default="msstft", metadata={"help": "discriminator"})
    discriminator_strategy: str = field(default="single", metadata={"help": "discriminator_strategy"})
    filters: int = field(default=32)
    disc_hop_lengths: Tuple[int, ...] = field(default=(256, 512, 128), metadata={"help": "disc hop lengths"})
    disc_win_lengths: Tuple[int, ...] = field(default=(1024, 2048, 512), metadata={"help": "disc win lengths"})
    disc_n_ffts: Tuple[int, ...] = field(default=(1024, 2048, 512), metadata={"help": "disc n_ffts"})
    update_disc_freq: int = field(default=3, metadata={"help": "update discriminator frequency, default is 3, it means every 3 steps update the discriminator"})
    disc_activation: str = field(default="LeakyReLU", metadata={"help": "discriminator activation function"})
    # loss
    n_mel_channels: int = field(default=32, metadata={"help": "n_mel channels"})



class EncodecGenerator(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """
    def __init__(self, cfg: EncodecConfig,task_cfg: EncodecTrainConfig) -> None:
        super().__init__()
        
        self.sample_rate = task_cfg.sample_rate
        self.channels = cfg.channels
        self.model_norm = cfg.model_norm
        self.normalize = cfg.audio_normalize
        self.overlap = cfg.overlap
        self.casual = cfg.casual
        self.encoder_ratios = cfg.encoder_ratios
        self.decoder_ratios = cfg.decoder_ratios
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder_ratios))
        
        self.encoder_dimension = cfg.encoder_dimension
        self.decoder_dimension = cfg.decoder_dimension
        
        self.target_bandwidths = cfg.target_bandwidths
        self.bandwidth = None
        self.lstm = cfg.lstm
        self.transformer = cfg.transformer
        self.n_head = cfg.n_head
        self.activation = cfg.activation
        self.activation_params = cfg.activation_params
        
        self.encoder = m.SEANetEncoder(
            channels=self.channels,
            dimension=self.encoder_dimension,
            norm=self.model_norm,
            causal=self.casual,
            ratios=self.encoder_ratios,
            lstm=self.lstm,
            transformer=self.transformer,
            n_head=self.n_head,
            activation=self.activation,
            activation_params=self.activation_params,
            n_residual_layers=cfg.encoder_n_residual_layers,
            n_filters=cfg.encoder_n_filters
        )
        self.decoder = m.SEANetDecoder(
            channels=self.channels,
            dimension=self.decoder_dimension,
            norm=self.model_norm,
            causal=self.casual,
            ratios=self.decoder_ratios,
            lstm=self.lstm,
            transformer=self.transformer,
            n_head=self.n_head,
            activation=self.activation,
            activation_params=self.activation_params,
            n_residual_layers=cfg.decoder_n_residual_layers,
            final_activation=cfg.final_activation,
            n_filters=cfg.decoder_n_filters
        )

        codebook_dim = cfg.codebook_dim
        requires_projection = codebook_dim != self.encoder.dimension*2
        
        self.random_select = cfg.random_select
        self.full_bandwidth = cfg.full_bandwidth
        
        n_q = cfg.n_q if cfg.n_q else int(1000 * self.target_bandwidths[-1] // (math.ceil(self.sample_rate / self.encoder.hop_length) * 10))
        bins = cfg.bins
        self.quantizer = qt.ResidualGaussionVectorQuantizer(
            dimension=self.encoder.dimension,
            codebook_dim=codebook_dim,
            n_q=n_q,
            bins=bins,
        )

        self.bits_per_codebook = int(math.log2(bins))
        
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, \
            "quantizer bins must be a power of 2."
        self.segment = None

        logger.info(
            f"quantizer setting: n_q: {n_q}, bins: {bins}, dimension: {self.quantizer.dimension}, random_select:{self.random_select}, requires_projection: {requires_projection} "
            f"encoder setting: dimension: {self.encoder_dimension}, ratios:{self.encoder_ratios}, LSTM: {self.lstm}, transformer: {self.transformer}, n_head: {self.n_head},"
            f"decoder setting: dimension: {self.decoder_dimension}, ratios:{self.decoder_ratios}, LSTM: {self.lstm}, transformer: {self.transformer}, n_head: {self.n_head},"
        )
    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))
    
    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. "
                             f"Select one of {self.target_bandwidths}.")
        self.bandwidth = bandwidth
    
    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length 
        if segment_length is None: #segment_length = 1*sample_rate
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride): # shift windows to choose data
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1] # tensor_cut or original
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        if self.training:
            return emb,scale
        
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)

        return codes, scale

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        if self.training:
            emb = codes
        else:
            codes = codes.transpose(0, 1)
            emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x: torch.Tensor, update_num) -> torch.Tensor:
        results = {}
        frames = self.encode(x)
        if self.training:
            codes = []
            n_qs = None
            # when training on multigpu, broadcast index to different devices
            index = torch.tensor(random.randint(0,len(self.target_bandwidths)-1),device=x.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(index, src=0)    
            if self.random_select:
                assert self.full_bandwidth == False, print("random_select and full_bandwidth can not be True at the same time")
                n_qs = torch.tensor(random.randint(0,self.quantizer.n_q-1),device=x.device)
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(n_qs, src=0)
            if self.full_bandwidth:
                n_qs = torch.tensor(0,device=x.device)
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(n_qs, src=0)
                    
            bw = self.target_bandwidths[index.item()]
            
            emb,scale = frames[0]
            quantized, _, _, _, commit_loss = self.quantizer(emb,self.frame_rate,bw,update_num,n_qs=n_qs)
            codes.append((quantized,scale))
            
            results["decoder_out"] = self.decode(codes)[:,:,:x.shape[-1]]
            results["loss_commit"] = commit_loss
            results["encode_frames"] = frames
            # results["expired_codes"] = expired_code
            
            return results
        else:
            results["decoder_out"] = self.decode(frames)[:,:,:x.shape[-1]]
            results["encode_frames"] = frames
            return results

class EncodecDiscriminator(nn.Module):
    def __init__(self, cfg: EncodecConfig) -> None:
        super().__init__()
        self.discriminator_name = cfg.discriminator
        if cfg.discriminator_strategy == "single":
            if cfg.discriminator == "msstft":
                # self.MSSTFTDiscriminator = MultiScaleSTFTDiscriminator(
                #     filters=cfg.filters,
                #     n_ffts=cfg.disc_n_ffts,
                #     hop_lengths=cfg.disc_hop_lengths,
                #     win_lengths=cfg.disc_win_lengths
                # )
                self.discriminator= MultiScaleSTFTDiscriminator(
                    filters=cfg.filters,
                    n_ffts=cfg.disc_n_ffts,
                    hop_lengths=cfg.disc_hop_lengths,
                    win_lengths=cfg.disc_win_lengths,
                    activation=cfg.disc_activation
                )
            elif cfg.discriminator == "mpd":
                self.discriminator= MultiPeriodDiscriminator()
            elif cfg.discriminator == "msp":
                self.discriminator= MultiScaleDiscriminator()
            elif cfg.discriminator == "mspec":
                self.discriminator= MultiResSpecDiscriminator()
        elif cfg.discriminator_strategy == "mix":        
            self.discriminators = {}
            self.discriminators["msstft"] = MultiScaleSTFTDiscriminator(
                    filters=cfg.filters,
                    n_ffts=cfg.disc_n_ffts,
                    hop_lengths=cfg.disc_hop_lengths,
                    win_lengths=cfg.disc_win_lengths,
                    activation=cfg.disc_activation
                )
            self.discriminators["mpd"] = MultiPeriodDiscriminator()
            self.discriminators["msp"] = MultiScaleDiscriminator()

    def forward(self,x):
        logits, fmaps = self.discriminator(x)
        return {
            f"{self.discriminator_name}_disc_logits": logits,
            f"{self.discriminator_name}_disc_fmaps": fmaps
        }
        # for key in self.discriminators:
        #     logits, fmaps = self.discriminators[key](x)
        #     results[f"{key}_disc_logits"] = logits
        #     results[f"{key}_disc_fmaps"] = fmaps
        # return results
        # if cfg.discriminator_strategy == "single":
        #     logits, fmaps = self.MSSTFTDiscriminator(x)
        #     return {
        #         "msstft_disc_logits": logits,
        #         "msstft_disc_fmaps": fmaps
        #     }
        # elif cfg.discriminator_strategy == "mix":
        #     logits_msstft, fmaps_msstft = self.discriminators["msstft"](x)
        #     logits_mpd, fmaps_mpd = self.discriminators["mpd"](x)
        #     logits_msp, fmaps_msp = self.discriminators["msp"](x)
        #     return {
        #         "msstft_disc_logits": logits_msstft,
        #         "msstft_disc_fmaps": fmaps_msstft,
        #         "mpd_disc_logits": logits_mpd,
        #         "mpd_disc_fmaps": fmaps_mpd,
        #         "msp_disc_logits": logits_msp,
        #         "msp_disc_fmaps": fmaps_msp
        #     }
    
@register_model("encodec",dataclass=EncodecConfig)
class EncodecModel(BaseFairseqModel):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """
    def __init__(self, cfg: EncodecConfig,task_cfg: EncodecTrainConfig) -> None:
        super().__init__()
        
        self.encodec_generator = EncodecGenerator(cfg=cfg,task_cfg=task_cfg)
        self.encodec_discriminator = EncodecDiscriminator(cfg=cfg)
        
        for p in self.encodec_discriminator.parameters():
            p.param_group = "discriminator"
        
        for p in self.encodec_generator.parameters():
            p.param_group = "generator"
        
        logger.info(f"Generator params: {sum(p.numel() for p in self.encodec_generator.parameters() if p.requires_grad)}")
        logger.info(f"Generator EnCoder params: {sum(p.numel() for p in self.encodec_generator.encoder.parameters() if p.requires_grad)}")
        logger.info(f"Generator DeCoder params: {sum(p.numel() for p in self.encodec_generator.decoder.parameters() if p.requires_grad)}")
        logger.info(f"Generator Quantizer params: {sum(p.numel() for p in self.encodec_generator.quantizer.parameters() if p.requires_grad)}")
        logger.info(f"Discriminator params: {sum(p.numel() for p in self.encodec_discriminator.parameters() if p.requires_grad)}")
            
        self.update_num = 0
        self.disriminator_model = cfg.discriminator
        self.not_train_discriminator_step = task_cfg.not_train_discriminator_step
        self.n_mel_channels = cfg.n_mel_channels
        self.sample_rate = self.encodec_generator.sample_rate
        self.update_disc_freq = cfg.update_disc_freq
        
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates

    def discrim_step(self, num_updates,not_train_discriminator_step):
        """get discrim step to judge update discriminator params

        Args:
            num_updates (_type_): _description_
            not_train_discriminator_step (_type_): _description_

        Returns:
            _type_: _description_
        """
        if (num_updates - not_train_discriminator_step) > 0 and (num_updates - not_train_discriminator_step) % self.update_disc_freq == 1:
        # 当step大于not_train_discriminator_step，每3步只更新1步disc
            return True
        else:
            return False
        
    def get_groups_for_update(self, num_updates):
        return "discriminator" if self.discrim_step(num_updates,self.not_train_discriminator_step) else "generator"
        
    def disc_loss(self, logits_real, logits_fake):
        """This function is used to compute the loss of the discriminator.
            l_d = \sum max(0, 1 - D_k(x)) + max(0, 1 + D_k(\hat x)) / K, K = disc.num_discriminators = len(logits_real) = len(logits_fake) = 3
        Args:
            logits_real (List[torch.Tensor]): logits_real = disc_model(input_wav)[0]
            logits_fake (List[torch.Tensor]): logits_fake = disc_model(model(input_wav)[0])[0]
        
        Returns:
            lossd: discriminator loss
        """
        l_d = 0.0
        for tt1 in range(len(logits_real)):
            l_d = l_d + torch.mean(F.relu(1-logits_real[tt1])) + torch.mean(F.relu(1+logits_fake[tt1]))
        l_d = l_d / len(logits_real)
        return l_d
    
    def adversarial_loss(self,logits_fake):
        l_g = 0.0
        for logit_fake in logits_fake:
            l_g = l_g + torch.mean(F.relu(1 - logit_fake))
            
        l_g = l_g / len(logits_fake)
        return l_g
            
    def feature_matching_loss(self,fmap_real,fmap_fake):
        l_feat = 0.0
        for tt1 in range(len(fmap_real)): # len(fmap_real) = 3
            for tt2 in range(len(fmap_real[tt1])): # len(fmap_real[tt1]) = 5
            # fmap_real is calculated from disc_model(x), when generator step, we don't need to update disc model -> fmap_real.detach()
                l_feat = l_feat + F.l1_loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2].detach())) # 可能就是修改的是数值部分
    
        KL_scale = len(fmap_real)*len(fmap_real[0]) # len(fmap_real) == len(fmap_fake) == len(logits_real) == len(logits_fake) == disc.num_discriminators == K
        
        return l_feat / KL_scale
    
    def reconstruction_loss_frequency_domain(self,input_sample,net_output,n_mel_channels):
        l_f = torch.tensor([0.0], device=input_sample.device, requires_grad=True)
        for i in range(5, 12): #e=5,...,11
            fft = Audio2Mel(n_fft=2 ** i,win_length=2 ** i, hop_length=(2 ** i) // 4, n_mel_channels=n_mel_channels, sampling_rate=self.sample_rate)
            l_f = l_f + F.l1_loss(fft(input_sample), fft(net_output),reduction="mean") + F.mse_loss(fft(input_sample), fft(net_output),reduction="mean")
        return l_f / 7
    
    def reconstruction_loss_time_domain(self,input_sample,net_output):
        l_t = torch.tensor([0.0], device=input_sample.device, requires_grad=True)
        l_t = F.l1_loss(input=net_output,target=input_sample,reduction="mean")
        return l_t 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        losses = {}
        d_step = self.discrim_step(self.update_num,self.not_train_discriminator_step)
        results = self.encodec_generator(x,update_num=self.update_num)
        disc_real_results = self.encodec_discriminator(x)
        
        if not d_step:
            # when not d_step, update generator
            disc_fake_results = self.encodec_discriminator(results["decoder_out"])
            
            losses["adversial_loss"] = self.adversarial_loss(
                logits_fake=disc_fake_results[f"{self.disriminator_model}_disc_logits"]
            )
            losses["feature_matching_loss"] = self.feature_matching_loss(
                fmap_real=disc_real_results[f"{self.disriminator_model}_disc_fmaps"],
                fmap_fake=disc_fake_results[f"{self.disriminator_model}_disc_fmaps"],
            )
            losses["frequency_domain_loss"] = self.reconstruction_loss_frequency_domain(
                input_sample=x,
                net_output=results["decoder_out"],
                n_mel_channels=self.n_mel_channels
            )
            losses["time_domain_loss"] = self.reconstruction_loss_time_domain(
                input_sample=x,
                net_output=results["decoder_out"]
            )

            if self.training:
                losses["commit_loss"] = results["loss_commit"]

        else:
            disc_fake_results = self.encodec_discriminator(results["decoder_out"].detach())
            
            losses["disc_loss"] = self.disc_loss(
                logits_real=disc_real_results[f"{self.disriminator_model}_disc_logits"],
                logits_fake=disc_fake_results[f"{self.disriminator_model}_disc_logits"],
            )
            
            with torch.no_grad():
                losses["adversial_loss"] = self.adversarial_loss(
                    logits_fake=disc_fake_results[f"{self.disriminator_model}_disc_logits"]
                )
                losses["feature_matching_loss"] = self.feature_matching_loss(
                    fmap_real=disc_real_results[f"{self.disriminator_model}_disc_fmaps"],
                    fmap_fake=disc_fake_results[f"{self.disriminator_model}_disc_fmaps"],
                )
                losses["frequency_domain_loss"] = self.reconstruction_loss_frequency_domain(
                    input_sample=x,
                    net_output=results["decoder_out"],
                    n_mel_channels=self.n_mel_channels,
                )
                losses["time_domain_loss"] = self.reconstruction_loss_time_domain(
                    input_sample=x,
                    net_output=results["decoder_out"],
                )

                if self.training:
                    losses["commit_loss"] = results["loss_commit"]
            
        return{
            "losses" : losses
        }
    
    @classmethod
    def build_model(cls, cfg: EncodecConfig, task: EncodecTrainTask):
        """Build a new model instance."""

        model = EncodecModel(cfg,task.cfg)
        return model    
