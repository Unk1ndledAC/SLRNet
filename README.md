# SLRNet: A Super Lightweight ResidualNet for Real Time Single Image Dehazing

This project has been used in a ongoing paper.
## 1.Overview
- This repo provides a light-weight, end-to-end single image dehazing network.
- Architecture: 3×3 conv + 5x (GhostModule + SEModule) + 5x ResBlocks + 3×3 conv, *computational cost: 4.9 GFLOPs, parameters: 100.6 K*.
- Loss: L1 + VGG-19 perceptual + FFT frequency loss. If only L1 loss is used, it will result in a performance degradation of about 2.2%.
- 5~10 epochs of training is recommanded for RESIDE-6K.
- On SOTS-outdoor (as test dataset) it reaches 34.33 dB PSNR / 0.9782 SSIM MAX and 25.21 dB PSNR / 0.9277 SSIM AVG after 10 epochs.
- On RESIDE-6K it costs 3s each training epoch and 0.8GB of VRAM in total on RTX 4070 Ti Super.
- The model size is 385KB.
- The average processing time for each image is about 8 ms on RTX 4070 Ti Super.
- This project integrates evaluations of metrics (time, PSNR, SSIM) in training and testing scripts, which can be removed for better performance.

## 2.Installation
```bash
# Python 3.10+ and CUDA 12.4 recommended
pip install -r requirements.txt
```

## 3.Usage
##### Specify the training-set path in `./scripts/train.py.`
##### Specify the test-set path in `./scripts/test.py.`
```bash
# 1. train
python -m scripts.train

# 2. test
python -m scripts.test
```


