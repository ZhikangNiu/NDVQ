# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

from .audio_to_mel import Audio2Mel


@dataclass
class EncodecCriterionConfig(FairseqDataclass):
    n_mel_channels: int = field(
        default=32,
        metadata={"help": "n_mel channels"}
    )
    loss_weights: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )


@register_criterion("encodec", dataclass=EncodecCriterionConfig)
class EncodecCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        n_mel_channels,
        loss_weights=None,
    ):
        super().__init__(task)
        self.n_mel_channels = n_mel_channels
        self.sample_rate = task.cfg.sample_rate
        self.loss_weights = loss_weights
        
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss = 0.0
        logging_output = {}

        input_sample = sample["net_input"]["source"]
        net_output = model(x=input_sample)
        # print(model.encodec_generator.quantizer.vq.layers[0]._codebook.embed)
        losses = net_output["losses"]

        if self.loss_weights is not None:
            for lk, lw in losses.items():
                if lk in self.loss_weights:
                    loss = loss + self.loss_weights[lk] * lw

        logging_output = {
            "loss": loss.item(),
            "reconstruction_loss_time_domain": losses["time_domain_loss"].item() * self.loss_weights["time_domain_loss"],
            "reconstruction_loss_frequency_domain": losses["frequency_domain_loss"].item() * self.loss_weights["frequency_domain_loss"],
            "adversial_loss": losses["adversial_loss"].item() * self.loss_weights["adversial_loss"],
            "feature_matching_loss": losses["feature_matching_loss"].item() * self.loss_weights["feature_matching_loss"],
            "sample_size": input_sample.shape[0],
            # "expired_codes": net_output["expired_codes"].item()
        }

        if losses.get("commit_loss") is not None:
            logging_output["commit_loss"] = losses["commit_loss"].item() * self.loss_weights["commit_loss"]
        if losses.get("disc_loss") is not None:
            loss = loss + losses["disc_loss"]
            logging_output["disc_loss"] = losses["disc_loss"].item()

        return loss*input_sample.shape[0], input_sample.shape[0], logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # expired_codes = sum(log.get("expired_codes", 0) for log in logging_outputs)
        # metrics.log_scalar(
        #     "expired_codes", expired_codes, sample_size, round=3
        # )
        metrics.log_scalar(
            "loss", loss_sum, sample_size, round=3
        )
        metrics.log_scalar(
            "reconstruction_loss_time_domain",
            (sum(log.get("reconstruction_loss_time_domain", 0) for log in logging_outputs)),
            sample_size, round=3
        )
        metrics.log_scalar(
            "reconstruction_loss_frequency_domain",
            (sum(log.get("reconstruction_loss_frequency_domain", 0) for log in logging_outputs)),
            sample_size, round=3
        )
        metrics.log_scalar(
            "adversial_loss",
            (sum(log.get("adversial_loss", 0) for log in logging_outputs)),
            sample_size, round=3
        )
        metrics.log_scalar(
            "feature_matching_loss",
            (sum(log.get("feature_matching_loss", 0) for log in logging_outputs)),
            sample_size, round=3
        )
        if len(logging_outputs) > 0 and logging_outputs[0].get("commit_loss"):
            commit_loss = sum(log.get("commit_loss", 0) for log in logging_outputs)
            if commit_loss >= 0:
                metrics.log_scalar(
                    "commitment_loss",
                    commit_loss,
                    sample_size, round=3
                )
        if len(logging_outputs) > 0 and logging_outputs[0].get("disc_loss"):
            metrics.log_scalar(
                "disc_loss",
                (sum(log.get("disc_loss", 0) for log in logging_outputs)),
                sample_size, round=3
            )

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
