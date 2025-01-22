# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Optional

import numpy as np
import torch

from fairseq.data.fairseq_dataset import FairseqDataset,data_utils

logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep):
    """load audio data

    Args:
        manifest_path (tsv): tsv file which save audio path
        max_keep (max keep): audio max keep size
        min_keep (int): audio min keep size

    Returns:
        _type_: _description_
    """
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes


class EncodecDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        crop_size=72000
    ):
        self.audio_root, self.audio_names, inds, tot, self.sizes = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.crop_size = crop_size

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        logger.info(
            f"shuffle={shuffle}, crop_size={crop_size}, "
            f"max_sample_size={self.max_sample_size}"
        )

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def __getitem__(self, index):
        wav = self.get_audio(index)
        wav = self.crop_to_max_size(wav,self.crop_size)
        return {"id": index, "source": wav}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        else:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
            return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        
        if len(samples) == 0:
            return {}
        
        audios = [s["source"][0] for s in samples]
        collated_audios = data_utils.collate_tokens(audios,pad_idx=0).unsqueeze(1)

        batch = {
            "ids": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "source": collated_audios
                },
        }
        return batch

    def size(self, index):
        if self.sizes[index] < self.crop_size:
            return self.sizes[index]
        else:
            return self.crop_size

    def num_tokens(self, index):
        return self.size(index)

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")
        
        return wav
