import os
import argparse
from train import train_dehaze
from test import test_dehaze
from data.dataset import get_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--train_root', type=str, default='./data/RESIDE-6K/train/')
    parser.add_argument('--test_root', type=str, default='./data/SOTS/outdoor')
    parser.add_argument('--model_path', type=str, default='./exp_SLR/best.pth')
    parser.add_argument('--out_dir', type=str, default='./exp_SLR/results')
    args = parser.parse_args()

    if args.mode == 'train':
        train_pairs = get_pairs(args.train_root)
        train_dehaze(train_pairs)
    else:
        test_pairs = get_pairs(args.test_root)
        test_dehaze(test_pairs, out_dir=args.out_dir, model_path=args.model_path)
