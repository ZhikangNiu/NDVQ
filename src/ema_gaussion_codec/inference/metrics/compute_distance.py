import argparse
import glob
import os
import numpy as np
import librosa
from losses import MelSpectrogramLoss, MultiScaleSTFTLoss
from audiotools import AudioSignal
from pathlib import Path
from tqdm import tqdm

def compute_mel_distance(ref_mel, deg_mel):
    # 计算Mel Distance
    mel_distance = np.mean(np.abs(ref_mel - deg_mel))
    return mel_distance

def compute_stft_distance(ref_stft, deg_stft):
    # 计算STFT Distance
    stft_distance = np.mean(np.abs(ref_stft - deg_stft))
    return stft_distance

def cal_mel_stft_distances(ref_dir, deg_dir):
    input_files = list(Path(deg_dir).rglob("*.wav"))
    ref_dir = Path(ref_dir)
    mel_loss = MelSpectrogramLoss()
    stft_loss = MultiScaleSTFTLoss()

    mel_distances = []
    stft_distances = []

    for deg_wav in tqdm(input_files):
        relative_path = deg_wav.relative_to(deg_dir)
        # ref_wav = ref_dir / relative_path.with_suffix('')
        if args.bw != 0.0:
            ref_wav = ref_dir / relative_path.parents[0] /deg_wav.name.replace(f"_bw{int(args.bw)}", "")
        else:
            ref_wav = ref_dir / relative_path
        # 读取WAV文件
        ref = AudioSignal(ref_wav)
        deg = AudioSignal(deg_wav)

        # 计算Mel频谱
        mel_dist = mel_loss(ref,deg)
        stft_dist = stft_loss(ref,deg)

        mel_distances.append(mel_dist)
        stft_distances.append(stft_dist)

    # 计算平均距离
    avg_mel_distance = np.mean(mel_distances)
    avg_stft_distance = np.mean(stft_distances)

    return avg_mel_distance, avg_stft_distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Mel Distance and STFT Distance measures.")

    parser.add_argument('-r', '--ref_dir', required=True, help="Reference wave folder.")
    parser.add_argument('-d', '--deg_dir', required=True, help="Degraded wave folder.")
    parser.add_argument('-b', '--bw', default=6.0, type=float, help="Bandwidth of the filter in kHz.")

    args = parser.parse_args()
    print(args)
    avg_mel_distance, avg_stft_distance = cal_mel_stft_distances(args.ref_dir, args.deg_dir)
    print(f"Average Mel Distance: {avg_mel_distance}")
    print(f"Average STFT Distance: {avg_stft_distance}")