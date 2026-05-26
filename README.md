# SLRNet: Super Lightweight Residual Network for Real-Time Image Dehazing

## 1.Overview
- This repo provides a light-weight, end-to-end image dehazing network.

![poster](architecture.png)

## 2.Installation
```bash
# Python 3.10+ and CUDA 12.4 recommended
pip install -r requirements.txt
```

## 3.Usage
1. Specify the training-set and test-set path.
```bash
# train & test
python -m SLRNet
```

## 4.Citation
This paper has been accepted as at **ICIC 2026** ([Link](http://www.ic-icc.cn/2026/)). [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20051973.svg)](https://doi.org/10.5281/zenodo.20051973)
```
@inproceedings{qu2026slrnet,
  title={SLRNet: Super Lightweight Residual Network for Real-Time Image Dehazing},
  author={Qu, Guanheng and Jiang, Fan and Liu, Jiangming},
  journal={},
  year={2026},
  address={Toronto, Canada},
  month={July},
  url={},
  doi = {10.5281/zenodo.20051973},
  note={Accepted for publication},
}
```
