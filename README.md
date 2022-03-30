# HandInteraction3D

[简体中文](README_zh.md)

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## Demo

![demo1](https://raw.githubusercontent.com/SheepHuan/yanghuan-images/main/img/demo1.gif)

The pretrained model will be released later.

## Framework

### Hand Detection

[Paddle Detection](https://github.com/PaddlePaddle/PaddleDetection)

### Keypoint Detection

1. [InterHand](https://github.com/facebookresearch/InterHand2.6M)
2. DualHand

## Introduce

Our repo is based on [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/). Below is the [related paper list](https://github.com/SheepHuan/PaperNote).

### Dataset

- [Interhands2.6M](https://github.com/facebookresearch/InterHand2.6M)

## Installation

```bash
# pip switch source
# Tsinghua source
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# Ali source
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Tencent source
pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
# Douban source
pip config set global.index-url http://pypi.douban.com/simple/
```

```bash
python create -n hands python=3.8
conda activate hands

# Install Paddle Inference Library
# https://paddle-inference-lib.bj.bcebos.com/2.2.1/python/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.2_cudnn8/paddlepaddle_gpu-2.2.1.post112-cp38-cp38-win_amd64.whl
pip install paddlepaddle_gpu-2.2.1.post112-cp38-cp38-win_amd64.whl

# Download TesnorRT https://developer.nvidia.com/nvidia-tensorrt-8x-download
# choose TesnorRT8.0
# Installatin guide: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip


# Install other packages
pip install -r requirements.txt

```

## Train

```bash
# docker
docker pull paddlepaddle/paddle:2.2.2-gpu-cuda11.2-cudnn8

# create docker environment


# multi GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus '0,1,2,3' train.py

# single GPU training
python  train.py

```

## Model Export

export ppyolo model.

```bash
cd path/to/PaddleDetection
python tools/export_model.py \
 --config configs/ppyolo/ppyolo_r18vd_voc.yml \
 --opt weights=output/ppyolo_r18vd_voc/best_model.pdparams \
 --output_dir path/to/HandInteraction/weights
```

## Inference

### Model Weights

Pretrained Model：None

Download the entire folder from the weight link and put it in the ` ${model_root_path} / weights ` folder.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/SheepHuan"><img src="https://avatars.githubusercontent.com/u/48245110?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Huan Yang</b></sub></a><br /><a href="#infra-SheepHuan" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/deyu-csu/HandInteraction3D/commits?author=SheepHuan" title="Tests">⚠️</a> <a href="https://github.com/deyu-csu/HandInteraction3D/commits?author=SheepHuan" title="Code">💻</a> <a href="https://github.com/deyu-csu/HandInteraction3D/issues?q=author%3ASheepHuan" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/Geeksun2018"><img src="https://avatars.githubusercontent.com/u/42086593?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Geeksun2018</b></sub></a><br /><a href="#infra-Geeksun2018" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/deyu-csu/HandInteraction3D/commits?author=Geeksun2018" title="Tests">⚠️</a> <a href="https://github.com/deyu-csu/HandInteraction3D/commits?author=Geeksun2018" title="Code">💻</a> <a href="#design-Geeksun2018" title="Design">🎨</a> <a href="https://github.com/deyu-csu/HandInteraction3D/issues?q=author%3AGeeksun2018" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/BangwenHe"><img src="https://avatars.githubusercontent.com/u/32662175?v=4?s=100" width="100px;" alt=""/><br /><sub><b>bangwhe</b></sub></a><br /><a href="https://github.com/deyu-csu/HandInteraction3D/commits?author=BangwenHe" title="Code">💻</a> <a href="#translation-BangwenHe" title="Translation">🌍</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
