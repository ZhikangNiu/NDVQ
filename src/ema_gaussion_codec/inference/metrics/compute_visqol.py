import argparse
import numpy as np

from audiotools import metrics
from audiotools import AudioSignal
from pathlib import Path
from tqdm import tqdm

def cal_sisdr_distances(ref_dir, deg_dir):
    input_files = list(Path(deg_dir).rglob("*.wav"))
    ref_dir = Path(ref_dir)

    loss = []

    for deg_wav in tqdm(input_files):
        relative_path = deg_wav.relative_to(deg_dir)
        # ref_wav = ref_dir / relative_path.with_suffix('')
        if args.bw != 0.0:
            ref_wav = ref_dir / relative_path.parents[0] /deg_wav.name.replace(f"_bw{int(args.bw)}", "")
        else:
            ref_wav = ref_dir / relative_path
        # 读取WAV文件
        ref = AudioSignal(ref_wav)
        if ref.duration <8.0 or ref.duration > 10.0:
            continue
        deg = AudioSignal(deg_wav)

        # 计算Mel频谱
        s = metrics.quality.visqol(ref,deg,"speech")
        loss.append(s.detach().numpy())

    # 计算平均距离
    avg_visqol = np.mean(loss)
    print(f"length of loss: {len(loss)}")
    return avg_visqol

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Mel Distance and STFT Distance measures.")

    parser.add_argument('-r', '--ref_dir', required=True, help="Reference wave folder.")
    parser.add_argument('-d', '--deg_dir', required=True, help="Degraded wave folder.")
    parser.add_argument('-b', '--bw', default=6.0, type=float, help="Bandwidth of the filter in kHz.")

    args = parser.parse_args()
    print(args)
    avg_visqol = cal_sisdr_distances(args.ref_dir, args.deg_dir)
    print(f"Average VISQOL: {avg_visqol}")