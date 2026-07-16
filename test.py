import os
import cv2
import time
import numpy as np
import torch
import tqdm
from skimage.metrics import structural_similarity as cal_ssim

from models.slrnet import SLRNet


def test_dehaze(test_pairs, out_dir='./exp_SLR/results', model_path='./exp_SLR/best.pth'):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SLRNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    all_psnr, all_ssim = [], []
    all_time = []

    for gt_path, hazy_path in tqdm.tqdm(test_pairs, ncols=100):
        I = cv2.imread(hazy_path)[:, :, ::-1]
        J_gt = cv2.imread(gt_path)[:, :, ::-1]
        if I.shape[:2] != J_gt.shape[:2]:
            I = cv2.resize(I, (J_gt.shape[1], J_gt.shape[0]))

        tensor = torch.from_numpy(I.copy()).permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        tic = time.time()
        J_hat = net(tensor)
        toc = time.time()

        J_hat_np = (J_hat * 255).cpu().squeeze(0).permute(1, 2, 0).detach().numpy()

        xxxx = os.path.splitext(os.path.basename(gt_path))[0]
        cv2.imwrite(os.path.join(out_dir, f'{xxxx}_dehaze.png'), J_hat_np[:, :, ::-1])

        mse = np.mean((J_hat_np - J_gt) ** 2)
        psnr = 20 * np.log10(255.) - 10 * np.log10(mse + 1e-10)
        ssim = cal_ssim(J_hat_np, J_gt, channel_axis=2, data_range=255)

        all_time.append((toc - tic) * 1000)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

    print(f'AVG: Time={np.mean(all_time):.2f}ms PSNR={np.mean(all_psnr):.2f} SSIM={np.mean(all_ssim):.4f}')
