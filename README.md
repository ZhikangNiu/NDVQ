# NDVQ
> Official repository of the IEEE SLT 2024 paper "NDVQ: Robust Neural Audio Codec with Normal Distribution-Based Vector Quantization"

>[!IMPORTANT] 
> I will release a clean codebase after refactoring and thoroughly testing the performance, so please be patient.

We introduced Normal Distribution Vector Quantization (NDVQ), which innovatively applies a distribution-based approach to vector quantization in audio codec. NDVQ’s adoption of normal distributions within the codebook enhances robustness and generalization capability, leading to a better reconstructed audio quality, especially at extremely low bandwidths. Our comparative analysis with EnCodec demonstrates NDVQ’s superior performance in audio compression tasks and downstream codec-based speech synthesis tasks, confirming its potential as a more resilient alternative to traditional VQ methods. While the real-world audio environment encompasses
speech, ambient sounds, and music, this investigation focused solely on speech, leaving other audio domains unexplored and thus limiting the applications. The challenge of developing a universal audio compression model based on our method represents a compelling
avenue for future research.

## NDVQ Architecture
![image](https://github.com/user-attachments/assets/0f719a1e-5864-4a7d-afda-5e83012c2876)

## Reconstruction Results at Various Bitrates
![image](https://github.com/user-attachments/assets/e5d1875c-3517-4077-ac97-b73b7a66c707)

## Install
### Conda Install
```shell
conda create -n fairseq python=3.9 -y 
conda activate fairseq
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install packaging editdistance gpustat wandb einops soundfile packaging librosa pandas

# install fairseq
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout 336c26a5
pip install --editable ./

# install apex
git clone https://github.com/NVIDIA/apex.git \
cd apex
git checkout 9263bc8 \
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Docker
You can download image from dockerhub by the following code
```
docker pull zkniu/fairseq:torch1.12-cu113-fairseq
```
More details about this image, you can check our Dockerfile in this repo.
## Training

## Inference

## Evaluation

## Acknowledge
1. We borrowed a lot of code from [encodec](https://github.com/facebookresearch/encodec)
2. We borrowed a lot of code from [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec/tree/main)
3. Thanks [fairseq](https://github.com/facebookresearch/fairseq) 

## Citation
Please cite the paper when referencing the NDVQ codebase and paper as:
```
@article{niu2024ndvq,
  title={NDVQ: Robust Neural Audio Codec with Normal Distribution-Based Vector Quantization},
  author={Niu, Zhikang and Chen, Sanyuan and Zhou, Long and Ma, Ziyang and Chen, Xie and Liu, Shujie},
  journal={arXiv preprint arXiv:2409.12717},
  year={2024}
}
```

## License
Our code is released under the MIT license 
