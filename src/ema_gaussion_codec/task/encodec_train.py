# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from fairseq.data import Dictionary
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask


logger = logging.getLogger(__name__)
@dataclass
class EncodecTrainConfig(FairseqDataclass):
    data: str = field(default="/data", metadata={"help": "path to data directory"})
    sample_rate: int = field(
        default=24_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )

    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    crop_size: int = field(
        default=72_000,
        metadata={"help": "always crop from the beginning if false"},
    )
    not_train_discriminator_step: int = field(
        default=10000,
        metadata={"help":"not update discriminator in 0~not_train_discriminator_step steps"}
    )
    

@register_task("encodec_training",dataclass=EncodecTrainConfig)  
class EncodecTrainTask(FairseqTask):

    cfg: EncodecTrainConfig

    def __init__(
        self, 
        cfg: FairseqDataclass
    ) -> None:
        super().__init__(cfg)
        
        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"EncodecTrainTask Config {cfg}")
        
        self.cfg = cfg
    
    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        self.datasets[split] = EncodecDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            crop_size=self.cfg.crop_size,
        )
        
    @property  
    def target_dictionary(self) -> Dictionary:  
        return None  
    
    def optimizer_step(self, optimizer, model, update_num):
        if hasattr(model, "get_groups_for_update"):
            groups = model.get_groups_for_update(update_num)
            optimizer.step(groups={groups})
        else:
            optimizer.step()
    
    @classmethod
    def setup_task(cls, cfg: EncodecTrainConfig, **kwargs) -> "EncodecTrainTask":
        return cls(cfg)