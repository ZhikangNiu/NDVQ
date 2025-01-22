# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""

import typing as tp
import warnings
import numpy as np
import math
from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
import random

from .. import distrib


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    """ema update parameter. moving_avg = moving_avg + (1-decay) * new
    Args:
        moving_avg (_type_): 
        new (_type_): update parameter
        decay (float): update rate
    """
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(embed):
    nn.init.kaiming_uniform_(embed)
    return embed

@torch.jit.script
def log_prob_density_fn(x,mu,logvar):
    # Reshape x for broadcasting  
    x = x.unsqueeze(1)  # Shape: [1800, 1024, 1]  
    
    # Calculate variance from log variance and reshape for broadcasting  
    var = torch.exp(logvar).unsqueeze(0)  # Shape: [1, 1024, 128]  
    mu = mu.unsqueeze(0)  # Shape: [1, 1024, 128]  
    logvar = logvar.unsqueeze(0)  # Shape: [1, 1024, 128]  
      
    # Constants for the log probability  
    log_two_pi = torch.log(torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype))  
    
    # Calculate the per-dimension log probability  
    # Broadcasting will take care of matching the dimensions  
    per_dim_log_prob = -0.5 * ((x - mu) ** 2) / var - 0.5 * logvar - 0.5 * log_two_pi  
    
    # Sum over the second dimension (features) to get the total log probability for each x and mu pair  
    log_prob = per_dim_log_prob.sum(dim=-1)  # Shape: [1800, 1024]  
    return log_prob

def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

class GaussionEMACodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.dim = dim
        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        self.embed = nn.Embedding(codebook_size, dim)
        

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))


    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return
        
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        var_init_embed = torch.zeros_like(embed)
        
        self.embed.weight.data[:,:self.dim//2].copy_(embed)
        self.embed.weight.data[:,self.dim//2:].copy_(var_init_embed)
        
        self.inited.data.copy_(torch.Tensor([True]))
        distrib.broadcast_tensors(self.buffers()) # FIXME: this is not working for some reason

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed.weight.data[:,:self.dim//2,]
        )
        return modified_codebook

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        self.expire_codes_num = sum(expired_codes)
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        modifered_codebook = self.replace_(batch_samples, mask=expired_codes)
        modifered_codebook_var = torch.zeros_like(modifered_codebook)
        
        self.embed.weight.data[:,:self.dim//2].copy_(modifered_codebook)
        self.embed.weight.data[:,self.dim//2:].copy_(modifered_codebook_var)
        distrib.broadcast_tensors(self.buffers()) # FIXME: this is not working for some reason

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x
    
    def quantize(self, x):
        mu,logvar = self.embed.weight.data[:,:self.dim//2],self.embed.weight.data[:,self.dim//2:]
        log_prob_res = log_prob_density_fn(x,mu,logvar)
        embed_ind = log_prob_res.max(dim=-1).indices # get the index of the closest embed

        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])
    
    def reparameterize(self, mu, logvar, num_updates, temp=1.0):
        if self.training:
            std = torch.exp(0.5*logvar) * temp
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

        # std = torch.exp(0.5*logvar) * temp
        # eps = torch.randn_like(std)
        # return mu + eps*std
    
    def dequantize(self, embed_ind, num_updates): # embed_ind b*time_step
        emb_vec = self.embed(embed_ind) # b*time_step*dim
        quantize = self.reparameterize(emb_vec[:,:,:self.dim//2],emb_vec[:,:,self.dim//2:],num_updates)
        return quantize,emb_vec[:,:,:self.dim//2],emb_vec[:,:,self.dim//2:]
        

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind,num_updates):
        quantize,mu,logvar = self.dequantize(embed_ind,num_updates)
        return quantize,mu,logvar

    def forward(self, x, num_updates):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x) # [Batch_size,time_step,dim] -> [(Batch_size,time_step),128]

        self.init_embed_(x) # to better initialize the codebook

        embed_ind = self.quantize(x) # get the index of the closest embed [(Batch_size,time_step)]
        embed_ind = self.postprocess_emb(embed_ind, shape) # [batch_size,time_step]
        quantize,mu,logvar = self.dequantize(embed_ind, num_updates)
            
        return quantize, embed_ind, mu, logvar, 0


class GaussionVectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size, the number of vectors in the codebook
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
                            the dimension of each vector in the codebook
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)
        _mean_dim = _logvar_dim = _codebook_dim // 2

        requires_projection = _codebook_dim != dim * 2
        self.project_in = (nn.Linear(dim, _mean_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_mean_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = GaussionEMACodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind,num_updates):
        quantize,mu,logvar = self._codebook.decode(embed_ind,num_updates)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x, num_updates):
        x = rearrange(x, "b d n -> b n d") # [2,128,32] -> [2,32,128]
        x = self.project_in(x)

        quantize, embed_ind ,mu,logvar, expire_code = self._codebook(x, num_updates)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=x.device, requires_grad=self.training)
        if self.training:
            if self.commitment_weight > 0:
                # commit_loss = F.mse_loss(mu.detach(),x) + F.mse_loss(quantize.detach(),x) + 0.25* F.mse_loss(quantize,x.detach()) + 0.25*F.mse_loss(mu,x.detach()) + 1e-5*torch.sum(F.mse_loss(logvar.exp(),torch.zeros_like(logvar),reduction='sum'))
                commit_loss = F.mse_loss(mu.detach(),x)  + 0.25*F.mse_loss(mu,x.detach()) + 1e-5*torch.sum(F.mse_loss(logvar.exp(),torch.zeros_like(logvar),reduction='sum'))
   
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss, expire_code

class ResidualGaussionVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [GaussionVectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None, num_updates: int= None):
        quantized_out = 0.0
        residual = x # x is encoder output emb
        all_losses = []
        all_indices = []
        quantized_list = []
        n_q = n_q or len(self.layers)
        print(len(self.layers[:n_q]))
        for i,layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss, _ = layer(residual,num_updates)
            quantized_list.append(quantized.detach()) # if you need to backward, we need to del detach()
            
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized # y^hat

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, quantized_list,out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices,num_updates=1)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices,num_updates=1)
            quantized_out = quantized_out + quantized
        return quantized_out


class ResidualGaussionVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
        if you want to know more information about RVQ, you can read the soundstream paper (https://arxiv.org/abs/2107.03312)
        Residual vector quantizer cascades N_q layers of VQ. 
        the algorithm is described as follows:
        **********************************************************************************
        Input: y = enc(x) the output of the encoder, vector quantizers Q_i for i = 1...N_q
        Output: the quantized y^hat
        
        y^hat <- 0 
        residual <- y
        for i=1 to N_q do
            y^hat += Q_i(residual)
            residual -= Q_i(residual)
        return y^hat

        **********************************************************************************
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        codebook_dim = None,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = ResidualGaussionVectorQuantization(
            dim=self.dimension,
            codebook_dim=codebook_dim,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    def forward(self, x: torch.Tensor, sample_rate: int, bandwidth: tp.Optional[float] = None, num_updates: int = 1,n_qs=None):
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        if n_qs is not None:
            n_q = n_qs
        # print(f"n_q: {n_q}, bw_per_q: {bw_per_q}")
        quantized, quantized_list, codes, commit_loss = self.vq(x, n_q=n_q,num_updates=num_updates)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return quantized,quantized_list,codes,bw,torch.mean(commit_loss)

    def get_num_quantizers_for_bandwidth(self, sample_rate: int, bandwidth: tp.Optional[float] = None) -> int:
        """Return n_q based on specified target bandwidth.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.:
            n_q = int(max(1, math.floor(bandwidth / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, sample_rate: int):
        """Return bandwidth per quantizer for a given input sample rate.
        """
        return math.log2(self.bins) * sample_rate / 1000

    def encode(self, x: torch.Tensor, sample_rate: int, bandwidth: tp.Optional[float] = None) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        codes = self.vq.encode(x, n_q=n_q) # vq.encode output -> out_indices
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        """
        quantized = self.vq.decode(codes) # vq.decode output -> quantized_out
        return quantized