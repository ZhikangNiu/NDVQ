FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

LABEL MAINTAINER="nzk020109@gmail.com"

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN set -x \
    && apt update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 vim ffmpeg sox\
    && apt install -y openssh-server 

RUN set -x \
    && apt -y install tmux \
    # && git config --global http.proxy http://127.0.0.1:7890 \
    && wget https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip\
    && unzip ninja-linux.zip -d /usr/local/bin/ \
    && rm -rf ninja-linux.zip \
    && update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force \
    # && git clone https://github.com/chxuan/vimplus.git ~/.vimplus \
    # && cd ~/.vimplus \
    # && ./install.sh
    && git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime \
    && sh ~/.vim_runtime/install_awesome_vimrc.sh \
    && echo "set number" >> ~/.vimrc

ENV PATH /opt/conda/bin:$PATH
ENV PATH /usr/local/cuda-11.3/bin:$PATH
ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH "/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
    
# install conda
RUN cd /root \
    && wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && . /root/.bashrc 

SHELL ["/bin/bash", "--login", "-c"]

# install fairseq
RUN conda create -n fairseq python=3.9 -y \
    && conda init bash \
    && conda activate fairseq \
    && conda install numpy matplotlib scipy -y \
    && echo "conda activate fairseq" >> ~/.bashrc \
    && conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y \
    && pip install packaging \
    && pip install editdistance \
    && pip install gpustat \
    && pip install tensorboard \
    && pip install wandb \
    && pip install einops \
    && pip install debugpy \
    && pip install soundfile \
    && apt-get install -y libsndfile1-dev \
    && cd / \
    && git clone https://github.com/facebookresearch/fairseq.git \
    && cd fairseq \
    && git checkout 336c26a5 \
    && pip install --editable ./ \
    && conda install -c conda-forge npy-append-array -y \
    && pip install librosa pandas sentencepiece \
    # install apex
    && cd / \
    && git clone https://github.com/NVIDIA/apex.git \
    && cd apex \
    && git checkout 9263bc8 \
    && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

ENV SHELL=/bin/bash
CMD [ "/bin/bash" ]