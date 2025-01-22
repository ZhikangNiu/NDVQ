import argparse
import glob
import os

import numpy as np
from pystoi import stoi
from scipy.io import wavfile
# from tqdm import tqdm
from pathlib import Path

def calculate_stoi(ref_dir, deg_dir):
    input_files = list(Path(deg_dir).rglob("*.wav"))
    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {ref_dir}")

    stoi_scores = []
    for deg_wav in input_files:
        relative_path = deg_wav.relative_to(deg_dir)
        if args.bw > 0:
            ref_wav = ref_dir / relative_path.parents[0] /deg_wav.name.replace(f'_bw{int(args.bw)}', '')
        else:
            ref_wav = ref_dir / relative_path
        rate, ref = wavfile.read(ref_wav)
        rate, deg = wavfile.read(deg_wav)
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        cur_stoi = stoi(ref, deg, rate, extended=False)
        stoi_scores.append(cur_stoi)

    return np.mean(stoi_scores)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute STOI measure")

    parser.add_argument(
        '-r', '--ref_dir', required=True, help="Reference wave folder.")
    parser.add_argument(
        '-d', '--deg_dir', required=True, help="Degraded wave folder.")
    parser.add_argument(
        '-b', '--bw', default=6, type=float, help="Bandwidth of the codec.")

    args = parser.parse_args()

    stoi_score = calculate_stoi(args.ref_dir, args.deg_dir)
    print(f"STOI: {stoi_score}")