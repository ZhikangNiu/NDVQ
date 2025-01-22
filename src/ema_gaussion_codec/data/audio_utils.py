# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import struct
import io
import json


def _group_to_batches_by_utters(buffer, sorted_idx_len_pair, batch_size,max_sample_size=None):
    batch_list = []

    single_batch = []
    for idx_len_pair in sorted_idx_len_pair:
        single_batch.append(buffer[idx_len_pair[0]])
        if len(single_batch) == batch_size:
            batch_list.append(single_batch)
            single_batch = []

    if len(single_batch) > 0:
        batch_list.append(single_batch)

    return batch_list


def _group_to_batches_by_frames(buffer, sorted_idx_len_pair, batch_size, max_sample_size=None):
    """Groups data into batches according to the number of frames.  
  
    Args:  
        buffer (list): A list containing all the data.  
        sorted_idx_len_pair (list of tuples): A list containing pairs of index and length, sorted by length.  
        batch_size (int): The maximum size of each batch.  
        max_sample_size (int, optional): The maximum size of each sample. If not specified, defaults to the batch size.  
  
    Returns:  
        list: A list containing all the batches.  
    """ 
    batch_list = [] # storage all btach with this list
    single_batch = [] # storage one single batch list
    frame_num_padded = 0
    if max_sample_size is None:
        max_sample_size = batch_size
    first_utt_len = min(max_sample_size, sorted_idx_len_pair[0][1]) # sorted_Idx_len_pair是传入的最长的，计算和max_sample_size最小的

    for idx_len_pair in sorted_idx_len_pair:
        frame_num_padded += first_utt_len
        if frame_num_padded > batch_size:
            if len(single_batch) > 0:
                batch_list.append(single_batch) # 存储single batch
                single_batch = []
                first_utt_len = min(max_sample_size, idx_len_pair[1])
                frame_num_padded = first_utt_len

        single_batch.append(buffer[idx_len_pair[0]])
    if len(single_batch) > 0:
        batch_list.append(single_batch)

    return batch_list


def _group_to_batches_by_frame_x_label(buffer, sorted_idx_len_pair, batch_size):
    batch_list = []

    single_batch = []
    frame_num_padded = 0

    max_lab_len = sorted_idx_len_pair[0][2] + 1
    max_utt_len = sorted_idx_len_pair[0][1]
    for idx_len_pair in sorted_idx_len_pair:
        if max_lab_len < idx_len_pair[2] + 1:
            max_lab_len = idx_len_pair[2] + 1
        frame_num_padded = max_utt_len * max_lab_len * (len(single_batch) )
        if frame_num_padded > batch_size:
            if len(single_batch) > 0:
                batch_list.append(single_batch)
                single_batch = []

                max_utt_len = idx_len_pair[1]
                max_lab_len = idx_len_pair[2] + 1

        single_batch.append(buffer[idx_len_pair[0]])

    if len(single_batch) > 0:
        batch_list.append(single_batch)

    return batch_list


class DataParser():
    def __init__(self):
        super().__init__()

    def _parse_data(self, data, data_type):
        if data_type.lower() == 'audio':
            parsed_data = self._parse_audio_data(data)
        elif data_type.lower() == 'info':
            parsed_data = self._parse_json_data(data)
        elif data_type.lower() == "feature":
            parsed_data = self._parse_feat_data(data)
        elif data_type.lower() == "label":
            parsed_data = self._parse_label_data(data)
        elif data_type.lower() == "codec":
            parsed_data = self._parse_codec_data(data)
        else:
            parsed_data = self._parse_string_data(data)
        return parsed_data

    def _parse_audio_data(self, data):
        byte_stream = io.BytesIO(data)
        import soundfile as sf
        with sf.SoundFile(byte_stream, 'r') as f:
            samples = f.read()
        return samples

    def _parse_label_data(self, data):  # one dimension numpy array

        label_pairs = []
        for i in range(int(len(data) / 4)):
            label = struct.unpack_from('<h', data, i * 4)[0]
            repeat_num = struct.unpack_from('<h', data, i * 4 + 2)[0]

            label_pairs.append((label, repeat_num))

        labels = []
        for label_pair in label_pairs:
            labels.extend([label_pair[0]] * label_pair[1])

        return " ".join(list(map(str, labels)))

    def _parse_json_data(self, data):
        str_data = str(data, 'utf-8')
        json_data = json.loads(str_data)

        return json_data

    def _parse_string_data(self, data):
        str_data = str(data, 'utf-8')
        return str_data

    def _parse_feat_data(self, data):
        feat = np.frombuffer(data, dtype=np.float32)
        feat = feat.reshape(-1, 80)

        return feat

    def _parse_codec_data(self, data):
        feat = np.frombuffer(data, dtype=np.int)
        return feat