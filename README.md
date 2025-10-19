# SLRNet: A Super Lightweight ResidualNet for Real Time Single Image Dehazing

This project has been used in a ongoing paper.

## 1.Overview
- This repo provides a light-weight, end-to-end single image dehazing network.
- Architecture: 5 ResBlocks + 2 3×3 conv, < 50 k parameters, occupying approximately 0.2GB of VRAM.
- Loss: L1 + VGG-19 perceptual + FFT frequency loss. If only L1 loss is used, it will result in a performance degradation of about 1%.
- On SOTS-outdoor it reaches ≈ 25 dB PSNR / 0.94 SSIM after only 2 epochs.
- On SOTS-indoor it costs 5s per epoch on RTX 4070 Ti Super.
- The average processing time for each image is about 10.39ms on RTX 4070 Ti Super.

## 2.Installation
```bash
    # Python 3.10+ and CUDA 12.4 recommended
    pip install -r requirements.txt
```

## 3.Usage
```bash
    # 1. train
    python -m scripts.train

    # 2. test
    python -m scripts.test
```
