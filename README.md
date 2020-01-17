# AlexNet_Base

Base code of AlexNet for CIFAR dataset on PyTorch  
You can train on any device (e.g. CPU / single GPU / multi GPU) and resume or test on any different device.

## Requirements

- python 3.5+
- pytorch 1.0+
- torchvision 0.4+

## How to train / evaluate AlexNet

### Usage

```
usage: main.py [-h] [-j N] [-b N] [--epochs N] [--lr LR] [--momentum M]
               [--wd W] [-p N] [--ckpt PATH] [-r] [-e] [-C]
               [-g GPUIDS [GPUIDS ...]] [--datapath PATH]
               DATA

positional arguments:
  DATA                  dataset: cifar10 | cifar100 (default: cifar10)

optional arguments:
  -h, --help            show this help message and exit
  -j N, --workers N     number of data loading workers (default: 8)
  -b N, --batch-size N  mini-batch size (default: 128), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel
  --epochs N            number of total epochs to run (default: 200)
  --lr LR, --learning-rate LR
                        initial learning rate (defualt: 0.1)
  --momentum M          momentum (default: 0.9)
  --wd W, --weight-decay W
                        weight decay (default: 5e-4)
  -p N, --print-freq N  print frequency (default: 50)
  --ckpt PATH           Path of checkpoint for resuming/testing or retraining
                        model (Default: none)
  -r, --resume          Resume model?
  -e, --evaluate        Test model?
  -C, --cuda            Use cuda?
  -g GPUIDS [GPUIDS ...], --gpuids GPUIDS [GPUIDS ...]
                        GPU IDs for using (Default: 0)
  --datapath PATH       where you want to load/save your dataset? (default:
                        ../data)
```

### Training

#### train on CPU

```shell
$ python3 main.py cifar10
```

#### train on Single GPU

```shell
$ python3 main.py cifar10 -C
```

##### if you want to use a specific GPU

```shell
$ python3 main.py cifar10 -C -g 2
```

or

```shell
$ CUDA_VISIBLE_DEVICES=2 python3 main.py cifar10 -C
```

#### train on multi GPU

```shell
$ python3 main.py cifar10 -C -g 0 1 2 3
```

### Evaluation

#### evaluate on CPU

```shell
$ python3 main.py cifar10 -e --ckpt ckpt_best.pth
```

#### evaluate on GPU

```shell
$ python3 main.py cifar10 -C -e --ckpt ckpt_best.pth
```

## References

- [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [AlexNet code on torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)