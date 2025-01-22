import argparse
import glob
import os
import numpy as np
import librosa
import torch
from pathlib import Path
from tqdm import tqdm

def cal_mos(deg_dir):
    input_files = list(Path(deg_dir).rglob("*.wav"))
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    mos_scores = []

    for deg_wav in tqdm(input_files):
        deg_wav, sr = librosa.load(deg_wav, sr=None)
        score = predictor(torch.from_numpy(deg_wav).unsqueeze(0), sr)
        
        mos_scores.append(score.detach().numpy())
        
    avg_score = np.mean(mos_scores)
    return avg_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Mel Distance and STFT Distance measures.")

    parser.add_argument('-d', '--deg_dir', required=True, help="Degraded wave folder.")
    parser.add_argument('-b', '--bw', default=6.0, type=float, help="Bandwidth of the filter in kHz.")

    args = parser.parse_args()
    print(args)
    avg_mos = cal_mos(args.deg_dir)
    print(f"Average Mos: {avg_mos}")