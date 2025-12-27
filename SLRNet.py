import os, glob, random, cv2, numpy as np, tqdm, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from skimage.metrics import structural_similarity as cal_ssim

lr_dehaze     = 1e-3
batch_size    = 32      
epochs        = 100    
crop_lr       = 256 
crop_length   = 5000
lambda_perc   = 0.2
lambda_freq   = 0.02
lambda_cont   = 0.01     
temperature   = 1.0    
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

class AdaptiveFeatureUnit(nn.Module):
    def __init__(self, ch, r=2, g=4):
        super().__init__()
        self.ch = ch
        self.split_ch = ch // r
        hidden_gate = max(4, ch // g)
        self.proj = nn.Conv2d(ch, self.split_ch, 1, bias=False)
        self.norm_proj = nn.BatchNorm2d(self.split_ch)
        self.refine = nn.Conv2d(self.split_ch, ch - self.split_ch, 3, padding=1, groups=self.split_ch, bias=False)
        self.norm_refine = nn.BatchNorm2d(ch - self.split_ch)
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_reduce = nn.Conv2d(self.split_ch, hidden_gate, 1, bias=False)
        self.gate_act = nn.GELU()
        self.gate_expand = nn.Conv2d(hidden_gate, ch, 1)
        self.gate_sig = nn.Sigmoid()
        
    def forward(self, x):
        z = self.norm_proj(self.proj(x))
        aux = self.norm_refine(self.refine(z))
        feat = torch.cat([z, aux], dim=1)
        gate = self.gate_pool(z)
        gate = self.gate_reduce(gate)
        gate = self.gate_act(gate)
        gate = self.gate_expand(gate)
        gate = self.gate_sig(gate)             
        return feat * gate            

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + res

class DehazeNet(nn.Module):
    def __init__(self, base_ch=32, n_blocks=5):
        super().__init__()
        self.in_proj = nn.Conv2d(3, base_ch, 3, padding=1)
        self.enhancer = AdaptiveFeatureUnit(base_ch)
        self.backbone = nn.ModuleList([ResBlock(base_ch) for _ in range(n_blocks)])
        self.refiner = AdaptiveFeatureUnit(base_ch)
        self.out_proj = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x):
        x = self.in_proj(x)
        early_feat = self.enhancer(x)
        feat = early_feat
        for block in self.backbone:
            feat = block(feat)
        late_feat = self.refiner(feat)
        fused = late_feat #+ early_feat
        out = self.out_proj(fused)
        return torch.clamp(out, 0.0, 1.0)

class VGGFeature(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights='DEFAULT').features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.parameters():
            p.requires_grad = False
        #self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        #x = self.normalize(x)
        return self.slice(x)

vgg_feat = VGGFeature().to(device)

def loss_perc(x, y): 
    return F.mse_loss(vgg_feat(x), vgg_feat(y))
    
def loss_freq(x, y): 
    return F.mse_loss(torch.fft.rfft2(x, norm='ortho').abs(),                 
                      torch.fft.rfft2(y, norm='ortho').abs())

def loss_cont(feat_pred, feat_gt):
    f_p = F.adaptive_avg_pool2d(feat_pred, (1, 1))
    f_g = F.adaptive_avg_pool2d(feat_gt, (1, 1))

    cos_sim = F.cosine_similarity(f_p, f_g, dim=1)
    loss = 1.0 - cos_sim.clamp(min=0.0, max=1.0)
    return loss.mean()

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
            I = I[:, top:top+self.crop, left:left+self.crop]
            J = J[:, top:top+self.crop, left:left+self.crop]

        if self.augment and random.random() > 0.5:
            I = I.flip(-1)
            J = J.flip(-1)

        return I, J

def train_dehaze(train_pairs, model_path='./exp_SLR'):
    os.makedirs(model_path, exist_ok=True)
    checkpoint_file = os.path.join(model_path, 'latest.pth')

    net = DehazeNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr_dehaze, weight_decay=1e-4)

    start_epoch = 1
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        net.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")

    train_set = HazyDataset(train_pairs, crop=crop_lr, length=crop_length, preload=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr_dehaze,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    if os.path.exists(checkpoint_file) and 'scheduler' in checkpoint:
        sched.load_state_dict(checkpoint['scheduler'])

    for epoch in range(start_epoch, epochs + 1):
        net.train()
        total_loss_sum = l1_sum = perc_sum = freq_sum = cont_sum = 0.0
        num_batches = 0

        pbar = tqdm.tqdm(train_loader, ncols=200, desc=f'Epoch {epoch}/{epochs}')
        for I, J in pbar:
            I, J = I.to(device), J.to(device)
            J_hat = net(I)

            l1_loss = F.l1_loss(J_hat, J)
            perc_loss = loss_perc(J_hat, J)
            freq_loss = loss_freq(J_hat, J)

            feat_J_hat = vgg_feat(J_hat)
            feat_J = vgg_feat(J).detach()
            cont_loss = loss_cont(feat_J_hat, feat_J)

            total_loss = (
                l1_loss +
                lambda_perc * perc_loss +
                lambda_freq * freq_loss +
                lambda_cont * cont_loss
            )

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            sched.step()

            total_loss_sum += total_loss.item()
            l1_sum += l1_loss.item()
            perc_sum += perc_loss.item()
            freq_sum += freq_loss.item()
            cont_sum += cont_loss.item()
            num_batches += 1

            pbar.set_postfix(
                total=total_loss.item(),
                l1=l1_loss.item(),
                perc=perc_loss.item(),
                freq=freq_loss.item(),
                cont=cont_loss.item()
            )

        torch.save(net.state_dict(), os.path.join(model_path, f'last.pth'))

        torch.save({
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': sched.state_dict(),
        }, checkpoint_file)

    torch.save(net.state_dict(), os.path.join(model_path, 'best.pth'))
    print(f'Training finished. Model saved to {model_path}')

def test_dehaze(test_pairs, out_dir='./exp_SLR/results', model_path='./exp_SLR/best.pth'):
    os.makedirs(out_dir, exist_ok=True)
    net = DehazeNet().to(device)
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
        
        all_time.append((toc - tic)*1000)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
       
    print(f'AVG: Time={np.mean(all_time):.2f}ms PSNR={np.mean(all_psnr):.2f} SSIM={np.mean(all_ssim):.4f}')

def get_pairs(root_dir):
    gt_dir   = os.path.join(root_dir, 'GT')
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

if __name__ == '__main__':
    train_pairs = get_pairs('./data/RESIDE_6K/train/') 
    train_dehaze(train_pairs)
    
    test_pairs = get_pairs('./data/SOTS/outdoor')  
    test_dehaze(test_pairs, model_path='./exp_SLR/best.pth')
    

