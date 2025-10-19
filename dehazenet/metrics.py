import torch
from skimage.metrics import structural_similarity as cal_ssim

def calc_psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    return (-10 * torch.log10(mse)).item()

def calc_ssim(x, y):
    x = (x * 255).clamp(0, 255).byte()
    y = (y * 255).clamp(0, 255).byte()
    ssim = 0.
    for i in range(x.size(0)):
        ssim += cal_ssim(x[i].permute(1, 2, 0).cpu().numpy(),
                         y[i].permute(1, 2, 0).cpu().numpy(),
                         channel_axis=2, data_range=255)
    return ssim / x.size(0)