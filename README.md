# PyTorch implementation of INIF (KSEM 2023)
This is an PyTorch implementation of the paper "Sparse-view CT Reconstruction via Implicit Neural Intensity Functions", KSEM 2023.

### Requirements
* Python 3.8
* PyTorch 1.7.1
* GPU memory >= 12GB

### Getting started
#### Prepare datasets
DL-sparse-view CT Challenge and TrueCT Reconstruction Challenge datasets, both can be freely downloaded from the [official website](https://www.aapm.org/GrandChallenge/) of AAPM Grand Challenge.

#### Train
* python mlp8256_multiscale_450_512_grad106.py --gpu 0 --sample 0 --expname mlp8256_450to512_grad_angle64 --sparse 64
