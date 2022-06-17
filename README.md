# A PyTorch-based implementation of Markonv

This repository contains the core code of Markonv neural layerand example for using Markonv. The scripts for Markonv layer and Markonv operator is in ./core. 

**Citation**: 

Jing-Yi Li, Yuhao Tan, Zheng-Yang Wen, Yu-Jian Kang, Yang Ding, and Ge Gao. Markonv: a novel convolutional layer with inter-positional correlations modeled. _bioRxiv_. doi: https://doi.org/10.1101/2022.06.09.495500

## Quick start

```python
from MarkonvCore import *
Markonvlayer = MarkonvV(kernel_length, kernel_number, channel_size, initKernelLen=None, channel_last=1, bias=False)
```
Parameters: 
```
kernel_length: Size of the convolutional kernel, max size of kernel in Markonv kernel.
kernel_number: Number of kernels in Markonv layer.
channel_size: Number of channels in input sequences, for example, DNA sequences encoded in one-hot encoding is 4.
initKernelLen: Size of mask, the init useful part in Markonv kernel.
channel_last: Whether the channel dimension in input sequence is the last dimension.
bias: If True, adds a learnable bias to the output. Default: False
```

### Code snippet
```python
import sys
sys.path.append("core")
from MarkonvCore import *
Markonvlayer = MarkonvV(kernel_length=10, kernel_number=16, channel_size=4).to("cuda:0")
input = torch.randn(20, 100, 4)
input = input.to("cuda:0")
output = Markonvlayer(input)
```

## Demo for training a Markonv-based neural network

0. Set a working space with conda

```bash
conda env create -f env_markonv.yaml
conda activate markonv
```

1. Decompress the dataset

```bash
cd external
tar -zxvf ./example.tar.gz
```


2. Training Markonv-based networks

```bash
cd ../scripts/demo
python3 torch_main.py
```
