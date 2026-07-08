import os
import glob
import random
import cv2
import numpy as np
import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HazyDataset(Dataset):
    def __init__(self, pairs, crop=256, length=None, augment=True, preload=False):
        self.crop = crop
        self.pairs = pairs
        self.trans = transforms.ToTensor()
        self.augment = augment
        self.preload = preload
        if length is not None and length > len(pairs):
            repeat = (length // len(pairs)) + 1
            self.pairs = (self.pairs * repeat)[:length]

        if preload:
            self.data = []
            for gp, hp in tqdm.tqdm(self.pairs, desc='Pre-loading'):
                j = cv2.imread(gp)[:, :, ::-1].copy()
                i = cv2.imread(hp)[:, :, ::-1].copy()
                if i.shape != j.shape:
                    i = cv2.resize(i, (j.shape[1], j.shape[0]))
                self.data.append((self.trans(i), self.trans(j)))
        else:
            self.data = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.data:
            I, J = self.data[idx]
        else:
            gp, hp = self.pairs[idx]
            j = cv2.imread(gp)[:, :, ::-1].copy()
            i = cv2.imread(hp)[:, :, ::-1].copy()

            if i.shape != j.shape:
                i = cv2.resize(i, (j.shape[1], j.shape[0]))
            I, J = self.trans(i), self.trans(j)

        if self.crop > 0:
            _, H, W = I.shape
            top = random.randint(0, H - self.crop)
            left = random.randint(0, W - self.crop)
            I = I[:, top:top + self.crop, left:left + self.crop]
            J = J[:, top:top + self.crop, left:left + self.crop]

        if self.augment and random.random() > 0.5:
            I = I.flip(-1)
            J = J.flip(-1)

        return I, J


def get_pairs(root_dir):
    gt_dir = os.path.join(root_dir, 'GT')
    hazy_dir = os.path.join(root_dir, 'hazy')
    gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.*')))
    hazy_dict = {}
    for hazy_path in glob.glob(os.path.join(hazy_dir, '*.*')):
        basename = os.path.splitext(os.path.basename(hazy_path))[0]
        key = basename.split('_')[0]
        hazy_dict.setdefault(key, []).append(hazy_path)

    pairs = []
    for gt_path in gt_list:
        key = os.path.splitext(os.path.basename(gt_path))[0]
        if key in hazy_dict:
            for hazy_path in hazy_dict[key]:
                pairs.append((gt_path, hazy_path))

    print(f'Total pairs: {len(pairs)}')
    return pairs


def get_pairs_(root_dir):
    gt_dir = os.path.join(root_dir, 'GT')
    hazy_dir = os.path.join(root_dir, 'hazy')
    gt_list = sorted(glob.glob(os.path.join(gt_dir, '*')))
    hazy_dict = {}
    for p in glob.glob(os.path.join(hazy_dir, '*')):
        key = os.path.splitext(os.path.basename(p))[0]
        hazy_dict.setdefault(key, []).append(p)

    pairs = []
    for gp in gt_list:
        key = os.path.splitext(os.path.basename(gp))[0]
        if key in hazy_dict:
            pairs.append((gp, random.choice(hazy_dict[key])))
    print(f'Total pairs: {len(pairs)}')
    return pairs
