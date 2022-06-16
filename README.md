# Example of how to train Markonv-based neural network
This repository is the Markonv neural layer built on PyTorch and example for using Markonv. the scripts contain Markonv layer and Markonv operator is in ./core. Below we show how to train a simple neural network based on Markonv layer.


## Requirements

### Python environment

```bash
conda env create -f env_markonv.yaml
conda activate markonv
```

## Quick start

```bash
from MarkonvCore import *
Markonvlayer = MarkonvV(kernel_length, kernel_number, channel_size, initKernelLen=None, channel_last=1, bias=False)
```
Parameters: 
```bash
kernel_length:Size of the convolving kernel, max size of kernel in Markonv kernel.
kernel_number: Number of kernels in Markonv layer.
channel_size: number of channel in input sequences, for example, DNA sequences encoded in one-hot encoding is 4.
initKernelLen: size of Mask, the init useful part in Markonv kernel.
channel_last: whether the channel dimension in input sequence is the last dimension.
bias:If True, adds a learnable bias to the output. Default: False
```

### example
```bash
Markonvlayer = MarkonvV(kernel_length=10, kernel_number=16, channel_size=4).to("cuda:0")
input = torch.randn(20, 100, 4)
input = input.to("cuda:0")
output = Markonvlayer(input)
```

## demo results
1. Decompress the dataset

```bash
cd external
tar -zxvf ./example.tar.gz
```


2. Training networks based Markonv

```bash
cd ../../scripts/demo
python3 torch_main.py
```



