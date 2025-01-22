# Inference Encodec and Evaluation

## reconstruct waveform
We can use main.py to reconstruct waveform, the main.py -h is listed as follows:

```shell
python main.py -h
usage: encodec [-h] [-b {1.5,3.0,6.0,12.0,24.0}] [-q] [-l] [-f] [-s DECOMPRESS_SUFFIX] [-r] [-m MODEL_NAME] [-c CHECKPOINT] [-d DEVICE] [--ext EXT] [--user_dir USER_DIR] input [output]

High fidelity neural audio codec. If input is a .ecdc, decompresses it. If input is .wav, compresses it. If output is also wav, do a compression/decompression cycle.

positional arguments:
  input                 Input file, whatever is supported by torchaudio on your system.
  output                Output file, otherwise inferred from input file.

optional arguments:
  -h, --help            show this help message and exit
  -b {1.5,3.0,6.0,12.0,24.0}, --bandwidth {1.5,3.0,6.0,12.0,24.0}
                        Target bandwidth (1.5, 3, 6, 12 or 24). 1.5 is not supported with --hq.
  -q, --hq              Use HQ stereo model operating on 48 kHz sampled audio.
  -l, --lm              Use a language model to reduce the model size (5x slower though).
  -f, --force           Overwrite output file if it exists.
  -s DECOMPRESS_SUFFIX, --decompress_suffix DECOMPRESS_SUFFIX
                        Suffix for the decompressed output file (if no output path specified)
  -r, --rescale         Automatically rescale the output to avoid clipping.
  -m MODEL_NAME, --model_name MODEL_NAME
                        support encodec_24khz,encodec_48khz,my_encodec
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        if use my_encodec, please input checkpoint
  -d DEVICE, --device DEVICE
  --ext EXT
  --user_dir USER_DIR, the encodec file absolute path

```
## evaluation
We can evaluate encodec with two objective metrics:

```shell
python cal_metrics.py
usage: cal_metrics.py [-h] -r REF_DIR -d DEG_DIR [-s SR] [-b BANDWIDTH] [-e EXT] [-o OUTPUT_RESULT_PATH]

Compute STOI and PESQ measure

optional arguments:
  -h, --help            show this help message and exit
  -r REF_DIR, --ref_dir REF_DIR
                        Reference wave folder.
  -d DEG_DIR, --deg_dir DEG_DIR
                        Degraded wave folder.
  -s SR, --sr SR        encodec sample rate.
  -b BANDWIDTH, --bandwidth BANDWIDTH
                        encodec bandwidth.
  -e EXT, --ext EXT     file extension
  -o OUTPUT_RESULT_PATH, --output_result_path OUTPUT_RESULT_PATH

```