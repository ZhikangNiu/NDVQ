from fairseq.utils import import_user_module
from fairseq import checkpoint_utils
import argparse
from encodec import EncodecModel

import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pylab as plt
import matplotlib;matplotlib.rc('font',family='Microsoft YaHei')
from sklearn import preprocessing

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

# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz().cpu()
# The number of codebooks used will be determined bythe bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.

models1, arg, task = checkpoint_utils.load_model_ensemble_and_task(["/data/gradient_rvq/gradient_vq_crop1s_lr3e-4_lt960_adam_change_vq_loss_add_every2vq/checkpoints/checkpoint607.pt"])  
# models2, arg, task = checkpoint_utils.load_model_ensemble_and_task(["/data/gaussion_vq/vae_codec_320x_8542_6e6d_lt960_adam_cosine_lr1e-4/checkpoints/checkpoint_47_226000.pt"])  
# models3, arg, task = checkpoint_utils.load_model_ensemble_and_task(["/data/gaussion_vq/vae_codec_320x_8542_6e6d_lt960_adam_cosine_lr1e-4_kl1e-5/checkpoints/checkpoint_45_217000.pt"]) 

model1 = models1[0].encodec_generator.cpu()
# model2 = models2[0].encodec_generator.cpu()
# model3 = models3[0].encodec_generator.cpu()

model.eval()
model1.eval()

encoedc_0_embed = model.quantizer.vq.layers[0]._codebook.embed.numpy()
ours_0_embed = model1.quantizer.vq.layers[0]._codebook.embed.weight.data[:,:128].numpy()
print(encoedc_0_embed.shape, ours_0_embed.shape)

B = preprocessing.minmax_scale(encoedc_0_embed, axis=1)   # 按列
OB = preprocessing.minmax_scale(ours_0_embed, axis=1)   # 按列

# B为标准化后的数据矩阵，按B的列聚类
julei = sch.linkage(B, metric='euclidean', method='single')

Ojulei = sch.linkage(OB, metric='euclidean', method='single')

# plt.figure(figsize=(100, 100))
for p in range(0,201,10):
    if p == 0:
        continue
    plt.title('Encodec Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')

    sch.dendrogram(
        julei,
        truncate_mode='level',
        p=p
    )
    plt.savefig(f'top{p}_level_encodec_cluster.png')
    plt.show()

    plt.title('GVQ Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    sch.dendrogram(
        Ojulei,
        truncate_mode='level',
        p=p
    )
    plt.savefig(f'top{p}_level_our_cluster.png')
    plt.show()