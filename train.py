import os
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from config import lr_dehaze, batch_size, epochs, crop_lr, crop_length, lambda_perc, lambda_freq, lambda_cont, num_workers
from data.dataset import HazyDataset
from models.slrnet import SLRNet
from losses.loss import get_vgg_feat, loss_perc, loss_freq, loss_cont


def train_dehaze(train_pairs, model_path='./exp_SLR'):
    os.makedirs(model_path, exist_ok=True)
    checkpoint_file = os.path.join(model_path, 'latest.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SLRNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr_dehaze, weight_decay=1e-4)
    vgg = get_vgg_feat(device)

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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr_dehaze, steps_per_epoch=len(train_loader), epochs=epochs
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

            feat_J_hat = vgg(J_hat)
            feat_J = vgg(J).detach()
            cont_loss = loss_cont(feat_J_hat, feat_J)

            total_loss = l1_loss + lambda_perc * perc_loss + lambda_freq * freq_loss + lambda_cont * cont_loss

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
