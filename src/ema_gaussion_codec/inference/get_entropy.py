from fairseq.utils import import_user_module
from fairseq import checkpoint_utils
import argparse
from encodec import EncodecModel
from encodec.utils import convert_audio
import os
import numpy as np
import torchaudio
import torch
# import dac

np.set_printoptions(threshold=np.inf)

def get_parser():
    parser = argparse.ArgumentParser(
        'encodec',
        description='High fidelity neural audio codec. '
                    'If input is a .ecdc, decompresses it. '
                    'If input is .wav, compresses it. If output is also wav, '
                    'do a compression/decompression cycle.')
    parser.add_argument(
        "--user_dir",
        default="//root/fairseq/examples/ema_gaussion_codec/")
    return parser
args = get_parser().parse_args()
import_user_module(args)

BAND_WIDTH = 6.0
CODEBOOKS = 8
# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz().cuda()
model.set_target_bandwidth(BAND_WIDTH)
model.eval()

# models1, arg, task = checkpoint_utils.load_model_ensemble_and_task(["/root/fairseq/examples/ema_gaussion_codec/inference/checkpoint607.pt"])  
# model1 = models1[0].encodec_generator.cuda()
# model1.set_target_bandwidth(BAND_WIDTH)

# model2 = dac.model.DAC.load("/root/.cache/descript/dac/weights_24khz_8kbps_0.0.4.pth").cuda()


# model1.eval()
# model2.eval()

number = 1000
path = []

with open("/root/fairseq/examples/ema_gaussion_codec/data_preprocess/lt960/train.tsv") as f:
    root = f.readline().strip()
    for line in f:
        items = line.strip().split("\t")
        path.append(os.path.join(root, items[0]))
f.close()

codes_bincount = {f"{i}" : [0]*1024 for i in range(CODEBOOKS)}   
# codes1_bincount = {f"{i}" : [0]*1024 for i in range(CODEBOOKS)}   
# codes2_bincount = {f"{i}" : [0]*1024 for i in range(CODEBOOKS)}   

e = {f"{i}":[] for i in range(CODEBOOKS)}
# e1 = {f"{i}":[] for i in range(CODEBOOKS)}
# e2 = {f"{i}":[] for i in range(CODEBOOKS)}
with torch.no_grad():
    for line in path[:number]:
        print(line)
        wav, sr = torchaudio.load(line)
        wav1 = wav.unsqueeze(0).cuda()
        
        code = model.encode(wav1)[0][0].squeeze(0) # torch.Size([1, 8, 716]) -> torch.Size([8, 716])   
        # code1 = model1.encode(wav1)[0][0].squeeze(0)
        # codes2 = model2.encode(wav1,24000)[1].squeeze(0)[:int(CODEBOOKS),:]
        for i in range(code.size(0)):
            codes_bincount[f"{i}"] = np.array(codes_bincount[f"{i}"]) + np.array(code[i,:].bincount(minlength=1024).tolist())
            # codes1_bincount[f"{i}"] = np.array(codes1_bincount[f"{i}"]) + np.array(code1[i,:].bincount(minlength=1024).tolist())
            # codes2_bincount[f"{i}"] = np.array(codes2_bincount[f"{i}"]) + np.array(codes2[i,:].bincount(minlength=1024).tolist())
            counts = (codes_bincount[f"{i}"] / codes_bincount[f"{i}"].sum()).clip(1e-10)
            # counts1 = (codes1_bincount[f"{i}"] / codes1_bincount[f"{i}"].sum()).clip(1e-10)
            # counts2 = (codes2_bincount[f"{i}"] / codes2_bincount[f"{i}"].sum()).clip(1e-10)
            entropy = -(counts * np.log2(counts)).sum()
            # entropy1 = -(counts1 * np.log2(counts1)).sum()
            # entropy2 = -(counts2 * np.log2(counts2)).sum()
            e[f"{i}"].append(entropy)
            # e1[f"{i}"].append(entropy1)
            # e2[f"{i}"].append(entropy2)
            # print(f"entropy of code {i}: {entropy}, entropy of GVQ code {i}: {entropy1}, entropy of DAC code {i}: {entropy2}")

for i in range(int(BAND_WIDTH)):
    with open(f"/root/fairseq/examples/ema_gaussion_codec/inference/lt-test_{i}_{number}_entropy.txt", "w") as f:
        f.write(f"{np.mean(e.get(f'{i}', [0]))}\n")
        # f.write(f"{np.mean(e1.get(f'{i}', [0]))}\n")
        # f.write(f"{np.mean(e2.get(f'{i}', [0]))}\n")
    f.close()
        
# print(f"average entropy of Encodec: {np.mean(e)}, average entropy of GVQ: {np.mean(e1)}, average entropy of DAC: {np.mean(e2)}")
#         code = model.encode(wav1)[0][0].squeeze(0) # torch.Size([1, 8, 716]) -> torch.Size([8, 716])   
# 统计字典里面每一个key对应的value中元素为0的索引
unused_encodec_code = {}
# unused_GVQ_code = {}
# unused_dac_code = {}
for z in range(int(BAND_WIDTH)):
    unused_encodec_code[f"{z}"] = [j for j, x in enumerate(codes_bincount[f"{z}"]) if x== 0]
    # unused_GVQ_code[f"{z}"] = [j for j, x in enumerate(codes1_bincount[f"{z}"]) if x== 0]
    # unused_dac_code[f"{z}"] = [j for j, x in enumerate(codes2_bincount[f"{z}"]) if x== 0]
    with open(f"/root/fairseq/examples/ema_gaussion_codec/inference/encodec_lt-test_{z}_{number}_bincount.txt", "w") as f:
        f.write(f"{codes_bincount.get(f'{i}', [0]*1024)}")
    f.close()
    # with open(f"/root/fairseq/examples/ema_gaussion_codec/inference/GVQ_lt-test_{z}_{number}_bincount.txt", "w") as f:
    #     f.write(f"{codes1_bincount.get(f'{i}', [0]*1024)}")
    # f.close()
    # with open(f"/root/fairseq/examples/ema_gaussion_codec/inference/DAC_lt-test_{z}_{number}_bincount.txt", "w") as f:
    #     f.write(f"{codes2_bincount.get(f'{i}', [0]*1024)}")
    # f.close()
    
    print(f"{z}: encodec unused code {unused_encodec_code[f'{z}']}")
    # print(f"{z}: GVQ unused code {unused_GVQ_code[f'{z}']}")
    # print(f"{z}: DAC unused code {unused_dac_code[f'{z}']}")
    


# for i in range(8):
#     with open(f"/root/fairseq/examples/ema_gaussion_codec/inference/encodec_{i}_bincount.txt", "w") as f:
#         f.write(f"{codes_bincount.get(f'{i}', [0]*1024)}")
#     f.close()
#     with open(f"/root/fairseq/examples/ema_gaussion_codec/inference/GVQ_{i}_bincount.txt", "w") as f:
#         f.write(f"{codes1_bincount.get(f'{i}', [0]*1024)}")
#     f.close()
# import matplotlib.pyplot as plt  
  
# # Number of subplots in each dimension  
# nrows = 2  
# ncols = 4  
  
# # Create a figure and an array of subplots with 2 rows and 4 columns  
# fig, axs = plt.subplots(nrows, ncols, figsize=(20, 10))  
  
# # Flatten the array of axes, which makes it easier to iterate over  
# axs = axs.flatten()  
  
# # Iterate over each codebook and plot the histogram  
# for i in range(CODEBOOKS):  
#     row = i // ncols  
#     col = i % ncols  
#     ax = axs[i]  
#     ax.bar(range(len(codes_bincount[f"{i}"])), codes_bincount[f"{i}"], color='blue')  
#     ax.set_title(f'Codebook {i} Usage')  
#     ax.set_xlabel('Code Index')  
#     ax.set_ylabel('Usage Count')  
  
# # Adjust layout to prevent overlap  
# plt.tight_layout()  
  
# # Save the figure to a file  
# plt.savefig('/root/fairseq/examples/ema_gaussion_codec/inference/figure.png')  
  
# # Optionally, display the plot  
# plt.show()  
import matplotlib.pyplot as plt  
  
# Number of subplots in each dimension  
nrows = 2  
ncols = 4  
  
# Create a figure and an array of subplots with 2 rows and 4 columns  
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 10))  
  
# Flatten the array of axes, which makes it easier to iterate over  
axs = axs.flatten()  
  
# Iterate over each codebook and plot the frequency histogram  
for i in range(CODEBOOKS):  
    # Calculate the frequency of each code index in the current codebook  
    total_counts = sum(codes_bincount[f"{i}"])  
    frequencies = [count / total_counts for count in codes_bincount[f"{i}"]]  
      
    # Calculate the number of unused codes in the current codebook  
    unused_codes_count = sum(1 for count in codes_bincount[f"{i}"] if count == 0)  
      
    ax = axs[i]  
    ax.bar(range(1024), frequencies, color='blue')  # Assuming 1024 is the length of the codebook  
    ax.set_title(f'Codebook {i} Frequency')  
    ax.set_xlabel('Code Index')  
    ax.set_ylabel('Frequency')  
      
    # Annotate the number of unused codes on the subplot  
    ax.text(0.95, 0.95, f'Unused codes: {unused_codes_count}',  
            verticalalignment='top', horizontalalignment='right',  
            transform=ax.transAxes,  
            color='red', fontsize=10)  
      
    # Set the y-axis limit to the max frequency of the current codebook  
    max_freq = max(frequencies)  
    ax.set_ylim(0, max_freq)  # Set the y-axis limit to the max frequency  
  
# Adjust layout to prevent overlap  
plt.tight_layout()  
  
# Save the figure to a file  
plt.savefig('/root/fairseq/examples/ema_gaussion_codec/inference/frequency_figure.png')  
  
# Optionally, display the plot  
plt.show()  
