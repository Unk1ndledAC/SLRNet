# SLRNet: A Super Lightweight ResidualNet for Real Time Image Dehazing

## 1.Overview
- This repo provides a light-weight, end-to-end single image dehazing network.

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

## 4.Dataset  
The dehazing dataset (RESIDE) is provided by Li et al.'s **Benchmarking Single-Image Dehazing and Beyond** [[arXiv](https://arxiv.org/abs/1712.04143)], licensed under the MIT License (data/LICENSE_MIT).












