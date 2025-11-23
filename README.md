# SLRNet: A Super Lightweight ResidualNet for Real Time Single Image Dehazing

This project has been used in a ongoing paper.
## 1.Overview
- This repo provides a light-weight, end-to-end single image dehazing network.
- Architecture: 3×3 conv + SEModule + GhostModule + 5x ResBlocks + GhostModule + SEModule + 3×3 conv, *computational cost: 4.79 GFLOPs, parameters: 96.7 K*.  
  ***Notice:*** A standalone GhostModule, SEModule, or ResBlock can each perform dehazing on its own; however, compared with the full model their PSNR and SSIM drop by 6.1 %~13.6 % and 1.6 %~8.2 %, respectively.  
  ***Computational cost and parameters among modules:***  
  **GhostModule**: 39.338 MFLOPs & 0.720K,  
  **SEModule**: 1.607 MFLOPs & 0.552K,  
  **ResBlock**: 0.924 GFLOPs & 18.496K
- Loss: L1 + VGG-19 perceptual + FFT frequency loss. If only L1 loss is used, it will result in a performance degradation of about 2.2%.
- This model was trained on the RESIDE-6K dataset, which comprises 3 000 indoor and 3 000 outdoor samples for training, and evaluated on the SOTS and OTS. After 10 epochs, it reaches 35.38 dB PSNR / 0.9852 SSIM MAX and 25.49 dB PSNR / 0.9379 SSIM AVG on SOTS-outdoor, 37.73 dB PSNR / 0.9904 SSIM MAX and 26.57 dB PSNR / 0.9521 SSIM AVG on OTS-BETA.
- 10 epochs of training is recommanded.
- It costs 15 s each training epoch and 2.9 GB of VRAM in total on RTX 4070 Ti Super.  
  ***Notice:*** The crop size (parameter `crop_lr` in `./scripts/train.py`) defaults to 128. Setting it to 64 cuts the per-epoch training time to about 4 s and reduces GPU memory usage to 0.9 GB, but it may introduce a 0.2 %~0.5 % performance drop and larger performance fluctuations.
- The model size is 394KB.
- The average inferring time for each image is about 2 ms on RTX 4070 Ti Super.
- This project integrates evaluations of metrics (time, PSNR, SSIM) in testing scripts, which can be removed for better performance.

## 2.Installation
```bash
# Python 3.10+ and CUDA 12.4 recommended
pip install -r requirements.txt
```

## 3.Usage
1. Specify the training-set path in `./scripts/train.py.`
2. Specify the test-set path in `./scripts/test.py.`
```bash
# 1. train
python -m scripts.train

# 2. test
python -m scripts.test
```







