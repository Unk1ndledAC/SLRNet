# SLRNet: A Super Lightweight ResidualNet for Real Time Single Image Dehazing

This project has been used in a ongoing paper.
## 1.Overview
- This repo provides a light-weight, end-to-end single image dehazing network.
- Architecture: 5 ResBlocks + 2 3×3 conv, 9.4 k parameters.
- Loss: L1 + VGG-19 perceptual + FFT frequency loss. If only L1 loss is used, it will result in a performance degradation of about 2.2%.
- 5 epochs of training is recommanded for RESIDE-6K.
- On SOTS-outdoor it reaches 31.17 dB PSNR / 0.9782 SSIM MAX and 23.33 dB PSNR / 0.9117 SSIM AVG after 2 epochs.
- On RESIDE-6K it costs 6s each training epoch and 0.7GB of VRAM in total on RTX 4070 Ti Super (80% train, 20% val).
- The model size is 375KB.
- The average inferring time for each image is about 1.12ms on RTX 4070 Ti Super.
- This project integrates evaluations of metrics (time, PSNR, SSIM) in training and testing scripts, which can be removed for better performance.

## 2.Installation
```bash
# Python 3.10+ and CUDA 12.4 recommended
pip install -r requirements.txt
```

## 3.Usage
##### Specify the training-set path in `./scripts/train.py.`
##### Specify the test-set path in `./scripts/test.py.`
##### Specify the ratio for splitting the training and validation sets in `./scripts/train.py.`
```bash
# 1. train
python -m scripts.train

# 2. test
python -m scripts.test
```
