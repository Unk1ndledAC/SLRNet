import os, glob, random, torch

def make_split(root, ratio=(0.8, 0.1, 0.1), seed=1234, save_txt=True):
    assert sum(ratio) == 1.0
    random.seed(seed)
    gt_list = sorted(glob.glob(os.path.join(root, 'gt', '*.*')))
    hazy_dict = {}
    for p in glob.glob(os.path.join(root, 'hazy', '*.*')):
        key = os.path.basename(p).split('.')[0]
        hazy_dict.setdefault(key, []).append(p)

    pairs = []
    for gp in gt_list:
        key = os.path.splitext(os.path.basename(gp))[0]
        if key in hazy_dict:
            pairs.append((gp, random.choice(hazy_dict[key])))

    random.shuffle(pairs)
    n_train = int(len(pairs) * ratio[0])
    n_val = int(len(pairs) * ratio[1])
    split = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:]
    }
    if save_txt:
        for phase, lst in split.items():
            with open(os.path.join(root, f'split_{phase}.txt'), 'w') as f:
                f.writelines([f'{g},{h}\n' for g, h in lst])
    return split


@torch.no_grad()
def validate(net, val_pairs, device, crop=0):
    from .dataset import HazyDataset
    from .metrics import calc_psnr, calc_ssim
    from torch.utils.data import DataLoader

    net.eval()
    val_set = HazyDataset(val_pairs, crop=crop)
    loader = DataLoader(val_set, batch_size=1, shuffle=False)
    psnr_sum = ssim_sum = n = 0
    for I, J in loader:
        I, J = I.to(device), J.to(device)
        J_hat = net(I).clamp(0, 1)
        psnr_sum += calc_psnr(J_hat, J)
        ssim_sum += calc_ssim(J_hat, J)
        n += 1
    return psnr_sum / n, ssim_sum / n