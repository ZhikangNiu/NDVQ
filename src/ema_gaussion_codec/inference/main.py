# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line for audio compression."""

import argparse
import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np

from compress import compress, decompress
from utils import convert_audio, save_audio
from fairseq.utils import import_user_module
from fairseq import checkpoint_utils
import random
from encodec import EncodecModel

SUFFIX = '.ecdc'


def get_parser():
    parser = argparse.ArgumentParser(
        'encodec',
        description='High fidelity neural audio codec. '
                    'If input is a .ecdc, decompresses it. '
                    'If input is .wav, compresses it. If output is also wav, '
                    'do a compression/decompression cycle.')
    parser.add_argument(
        'input', type=Path,
        help='Input file, whatever is supported by torchaudio on your system.')
    parser.add_argument(
        'output', type=Path, nargs='?',
        help='Output file, otherwise inferred from input file.')
    parser.add_argument(
        '-b', '--bandwidth', type=float, default=6, choices=[1.0,1.5, 3., 6., 12., 24.],
        help='Target bandwidth (1.5, 3, 6, 12 or 24). 1.5 is not supported with --hq.')
    parser.add_argument(
        '-q', '--hq', action='store_true',
        help='Use HQ stereo model operating on 48 kHz sampled audio.')
    parser.add_argument(
        '-l', '--lm', action='store_true',
        help='Use a language model to reduce the model size (5x slower though).')
    parser.add_argument(
        '-f', '--force', action='store_true',
        help='Overwrite output file if it exists.')
    parser.add_argument(
        '-s', '--decompress_suffix', type=str, default='_decompressed',
        help='Suffix for the decompressed output file (if no output path specified)')
    parser.add_argument(
        '-r', '--rescale', action='store_true',
        help='Automatically rescale the output to avoid clipping.')
    parser.add_argument(
        "--og", action='store_true',
        help="use original model"
    )
    parser.add_argument(
        '-m','--model_name', type=str, default='encodec_24khz',
        help='support encodec_24khz,encodec_48khz,my_encodec')
    parser.add_argument(
        '-c','--checkpoint', type=str, 
        help='if use my_encodec, please input checkpoint')
    parser.add_argument(
        "-d","--device",type=str,
        default="cuda"
    )
    parser.add_argument(
        "--ext",type=str,
        default="wav"
    )
    parser.add_argument(
        "--audio_num",type=int,
        default=500
    )
    parser.add_argument(
        "--seed",type=int,
        default=42
    )
    parser.add_argument(
        "--user_dir",
        default="/root/fairseq/examples/ema_gaussion_codec")
    return parser

def set_seed(seed):
    """set seed

    Args:
        seed (int): seed number
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def fatal(*args):
    print(*args, file=sys.stderr)
    sys.exit(1)


def check_output_exists(args):
    if not args.output.parent.exists():
        fatal(f"Output folder for {args.output} does not exist.")
    if args.output.exists() and not args.force:
        fatal(f"Output file {args.output} exist. Use -f / --force to overwrite.")


def check_clipping(wav, args):
    if args.rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


def main(args,model):
    if args.input.suffix.lower() == SUFFIX:
        # Decompression
        if args.output is None:
            args.output = args.input.with_name(args.input.stem + args.decompress_suffix).with_suffix('.wav')
        elif args.output.suffix.lower() != '.wav':
            fatal("Output extension must be .wav")
        check_output_exists(args)
        out, out_sample_rate = decompress(args.input.read_bytes(),device=args.device())
        check_clipping(out, args)
        save_audio(out, args.output, out_sample_rate, rescale=args.rescale)
    else:
        # Compression
        if args.output is None:
            args.output = args.input.with_suffix(SUFFIX)
        elif args.output.suffix.lower() not in [SUFFIX, '.wav']:
            fatal(f"Output extension must be .wav or {SUFFIX}")
        check_output_exists(args)

        wav, sr = torchaudio.load(args.input)
        if sr != model.sample_rate:
            wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.to(args.device)
        compressed = compress(model, wav, use_lm=args.lm)
        if args.output.suffix.lower() == SUFFIX:
            args.output.write_bytes(compressed)
        else:
            # Directly run decompression stage
            assert args.output.suffix.lower() == '.wav'
            out, out_sample_rate = decompress(model,compressed,device=args.device)
            check_clipping(out, args)
            save_audio(out, args.output, out_sample_rate, rescale=args.rescale)

def cli_main(args):
    if args.og:
        model = EncodecModel.encodec_model_24khz().cuda()
    else:
        models, arg, task = checkpoint_utils.load_model_ensemble_and_task([args.checkpoint])  
        model = models[0].encodec_generator.cuda()
    model.eval()
    model = model.to(args.device)
    model.set_target_bandwidth(args.bandwidth)
    
    if args.input.is_dir():
        output_root = args.output
        input_root = args.input
        if not output_root.exists():
            output_root.mkdir(parents=True)
        wav_list = list(args.input.glob(f'**/*.{args.ext}'))
        if args.audio_num < len(wav_list) and args.audio_num > 0:
            wav_list = random.sample(wav_list, args.audio_num)
        elif args.audio_num == -1 or args.audio_num > len(wav_list):
            wav_list = wav_list
        for wav in wav_list:
            relative_path = wav.relative_to(input_root)
            args.input = wav
            output_path = output_root.joinpath(relative_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output = output_path.with_name(output_path.stem + f"_bw{int(args.bandwidth)}.wav")
            try:
                main(args,model)
            except Exception as e:
                print(f"Error processing {wav}: {e}")
    elif args.input.is_file():
        main(args,model)

if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)
    import_user_module(args)
    set_seed(args.seed)
    if not args.input.exists():
        fatal(f"Input file {args.input} does not exist.")
    cli_main(args)
