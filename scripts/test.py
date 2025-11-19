import os, cv2, torch, numpy as np, tqdm, time
from dehazenet import DehazeNet
from skimage.metrics import structural_similarity as cal_ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_dehaze(test_pairs, out_dir, model_path='weights/best.pth'):
    os.makedirs(out_dir, exist_ok=True)
    net = DehazeNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    all_psnr, all_ssim = [], []
    time_list = []

    for gt_path, hazy_path in tqdm.tqdm(test_pairs, ncols=80):
        tic = time.time()
        I = cv2.imread(hazy_path)[:, :, ::-1]
        J_gt = cv2.imread(gt_path)[:, :, ::-1]
        if I.shape[:2] != J_gt.shape[:2]:
            I = cv2.resize(I, (J_gt.shape[1], J_gt.shape[0]))

        h, w = I.shape[:2]
        new_h, new_w = (h // 64) * 64, (w // 64) * 64
        if (new_h, new_w) != (h, w):
            I = cv2.resize(I, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            J_gt = cv2.resize(J_gt, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        tensor = torch.from_numpy(I).permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        
        J_hat = net(tensor).clamp(0, 1)
        toc = time.time()
        J_hat_np = (J_hat * 255).cpu().squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.uint8)

        xxxx = os.path.splitext(os.path.basename(gt_path))[0]
        cv2.imwrite(os.path.join(out_dir, f'{xxxx}_dehaze.png'), J_hat_np[:, :, ::-1])

        mse = np.mean((J_hat_np.astype(np.float32) - J_gt.astype(np.float32)) ** 2)
        psnr = 20 * np.log10(255.) - 10 * np.log10(mse + 1e-10)
        ssim = cal_ssim(J_hat_np, J_gt, channel_axis=2, data_range=255)
        
        time_list.append((toc - tic)*1000)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

    print(f'[Test] Time per image: Mean={np.mean(time_list):.2f} ms | 'f'Max={np.max(time_list):.2f} ms | Min={np.min(time_list):.2f} ms')

    print(f'[Test] AVG: PSNR={np.mean(all_psnr):.2f} SSIM={np.mean(all_ssim):.4f}')
    print(f'[Test] MAX: PSNR={np.max(all_psnr):.2f} SSIM={np.max(all_ssim):.4f}')
    print(f'[Test] MIN: PSNR={np.min(all_psnr):.2f} SSIM={np.min(all_ssim):.4f}')
    print(f'Images saved to {out_dir}')

if __name__ == '__main__':
    from dehazenet.utils import get_pairs
    test_pairs  = get_pairs('./data/RESIDE_6K/test')
    test_dehaze(test_pairs, out_dir='./result')
