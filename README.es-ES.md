

# NDVQ
> Repositorio oficial del artículo de IEEE SLT 2024 "NDVQ: Codificador de audio neural robusto con cuantización vectorial basada en distribución normal"

>[!IMPORTANT] 
> Lamento sinceramente no haber tenido mucho tiempo recientemente para limpiar mi código, por lo que compartiré una versión experimental y desordenada del repositorio que he estado utilizando.

Presentamos la Cuantización Vectorial de Distribución Normal (NDVQ), que aplica de manera innovadora un enfoque basado en distribuciones a la cuantización vectorial en códecs de audio. La incorporación de distribuciones normales en el codebook por parte de NDVQ mejora la robustez y la capacidad de generalización, lo que conduce a una mayor calidad de audio reconstruido, especialmente a anchos de banda extremadamente bajos. Nuestro análisis comparativo con EnCodec demuestra el rendimiento superior de NDVQ en tareas de compresión de audio y en tareas de síntesis de voz basadas en códecs, confirmando su potencial como una alternativa más resistente a los métodos de cuantización vectorial (VQ) tradicionales. Aunque el entorno de audio del mundo real abarca la voz, los sonidos ambientales y la música, esta investigación se centró únicamente en la voz, dejando otros dominios de audio sin explorar y limitando así las aplicaciones. El desafío de desarrollar un modelo universal de compresión de audio basado en nuestro método representa una vía prometedora para la investigación futura.

## Arquitectura de NDVQ
![image](https://github.com/user-attachments/assets/0f719a1e-5864-4a7d-afda-5e83012c2876)

## Resultados de Reconstrucción a Diferentes Tasas de Bits
![image](https://github.com/user-attachments/assets/e5d1875c-3517-4077-ac97-b73b7a66c707)

## Instalación
### Instalación con Conda
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

# clone repo
git clone https://github.com/ZhikangNiu/NDVQ.git
mv src/ema_gaussion_codec fairseq/examples
```

### Docker
Puede descargar la imagen desde DockerHub con el siguiente comando:
```
docker pull zkniu/fairseq:torch1.12-cu113-fairseq
```
Para más detalles sobre esta imagen, puede consultar nuestro Dockerfile en este repositorio.
## Entrenamiento
### Preparación de los datos
See examples/ema_gaussion_codec/scripts/wav2vec_manifest.py
### Entrenar
```
cd examples/ema_gaussion_codec/scripts/train
bash train.sh [CONFIG_NAME]
```
## Inferencia y Evaluación
```
cd examples/ema_gaussion_codec/inference
bash compute_metrics.sh [ref_dir] [gen_dir] [bw]
```
## Agradecimientos
1. Hemos tomado mucho código de [encodec](https://github.com/facebookresearch/encodec)
2. Hemos tomado mucho código de [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec/tree/main)
3. Gracias a [fairseq](https://github.com/facebookresearch/fairseq) 

## Cita
Por favor, cite el artículo al hacer referencia al código y al artículo de NDVQ de la siguiente manera:
```
@article{niu2024ndvq,
  title={NDVQ: Robust Neural Audio Codec with Normal Distribution-Based Vector Quantization},
  author={Niu, Zhikang and Chen, Sanyuan and Zhou, Long and Ma, Ziyang and Chen, Xie and Liu, Shujie},
  journal={arXiv preprint arXiv:2409.12717},
  year={2024}
}
```

## Licencia
Nuestro código se distribuye bajo la licencia MIT
