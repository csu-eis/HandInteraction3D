# HandInteraction3D

## Demo

![demo1](https://raw.githubusercontent.com/SheepHuan/yanghuan-images/main/img/demo1.gif)



预训练模型后续释出。。。

## Framework

### Hand Detection

[Paddle Detection](https://github.com/PaddlePaddle/PaddleDetection)

### Keypoint Detection

1. [InterHand](https://github.com/facebookresearch/InterHand2.6M)
2. DualHand

## Introduce

基于[InterHand2.6M](https://mks0601.github.io/InterHand2.6M/)的改进，此处为相关[文章列表](https://github.com/SheepHuan/PaperNote)。

### Dataset

- [Interhands2.6M](https://github.com/facebookresearch/InterHand2.6M)

## Installation

```bash
# pip 换源
# 清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 腾讯源
pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
# 豆瓣源
pip config set global.index-url http://pypi.douban.com/simple/
```



```bash
python create -n hands python=3.8
conda activate hands

# Install Paddle预测库 
# https://paddle-inference-lib.bj.bcebos.com/2.2.1/python/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.2_cudnn8/paddlepaddle_gpu-2.2.1.post112-cp38-cp38-win_amd64.whl
pip install paddlepaddle_gpu-2.2.1.post112-cp38-cp38-win_amd64.whl

# 下载 TesnorRT https://developer.nvidia.com/nvidia-tensorrt-8x-download
# 选择TesnorRT8.0版本
# 安装指南 https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip


# Install other packages
pip install -r requirements.txt

```

## Train
```bash
# docker
docker pull paddlepaddle/paddle:2.2.2-gpu-cuda11.2-cudnn8

# 创建docker环境



#多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus '0,1,2,3' train.py

#单卡训练
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

模型权重链接：None

从权重链接上下载整个文件夹后，放到`${model_root_path}/weights`文件夹中。
