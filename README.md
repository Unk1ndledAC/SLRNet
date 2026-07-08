# SLRNet: Super Lightweight Residual Network for Real-Time Image Dehazing

[![DOI](https://img.shields.io/badge/DOI-zenodo.20051973-blue.svg)](https://doi.org/10.5281/zenodo.20051973)

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
This paper has been accepted by **ICIC 2026** ([Link](http://www.ic-icc.cn/2026/)).
```
@inproceedings{qu2026slrnet,
  title={SLRNet: Super Lightweight Residual Network for Real-Time Image Dehazing},
  author={Qu, Guanheng and Jiang, Fan and Liu, Jiangming},
  journal={},
  year={2026},
  address={Toronto, Canada},
  month={July},
  url={},
  doi = {},
  note={Accepted for publication},
}
```
or
```
@software{Qu_SLRNet_2026,
  author = {Qu, Guanheng and Fan, Jiang and Liu, Jiangming},
  doi = {10.5281/zenodo.20051973},
  month = may,
  title = {{SLRNet}},
  url = {https://github.com/Unk1ndledAC/SLRNet},
  version = {1.0.0},
  year = {2026}
}
```
