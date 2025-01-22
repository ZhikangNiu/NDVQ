import argparse
import glob
import os

import scipy.signal as signal
from pesq import pesq,cypesq
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path

def cal_pesq(ref_dir, deg_dir):
    # input_files = glob.glob(f"{deg_dir}/*.wav")
    input_files = list(Path(deg_dir).rglob("*.wav"))
    # print(len(input_files))
    ref_dir = Path(ref_dir)
    nb_pesq_scores = 0.0
    wb_pesq_scores = 0.0
    for deg_wav in input_files:
        # ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
        relative_path = deg_wav.relative_to(deg_dir)
        # print(releative_path)
        # ref_wav = ref_dir / releative_path
        if args.bw > 0:
            ref_wav = ref_dir / relative_path.parents[0] /deg_wav.name.replace(f'_bw{int(args.bw)}', '')
        else:
            ref_wav = ref_dir / relative_path
        # print(ref_wav)
        # print(deg_wav)
        ref_rate, ref = wavfile.read(ref_wav)
        deg_rate, deg = wavfile.read(deg_wav)
        # if len(deg.shape) != 1:
            # deg = deg.mean(axis=1)
        # print(ref.shape, deg.shape)
        if ref_rate != 16000:
            ref = signal.resample(ref, 16000)
        if deg_rate != 16000:
            deg = signal.resample(deg, 16000)

        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        try:
            nb_pesq_scores += pesq(16000, ref, deg, 'nb')
            wb_pesq_scores += pesq(16000, ref, deg, 'wb')
        except cypesq.NoUtterancesError:
            nb_pesq_scores+=0
            wb_pesq_scores+=0

    return nb_pesq_scores / len(input_files), wb_pesq_scores / len(input_files)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute PESQ measure.")

    parser.add_argument(
        '-r', '--ref_dir', required=True, help="Reference wave folder.")
    parser.add_argument(
        '-d', '--deg_dir', required=True, help="Degraded wave folder.")
    parser.add_argument(
        '-b', "--bw", default=6, type=float, help="Bandwidth of the codec.")

    args = parser.parse_args()
    print(args)
    nb_score, wb_score = cal_pesq(args.ref_dir, args.deg_dir)
    print(f"NB PESQ: {nb_score}")
    print(f"WB PESQ: {wb_score}")