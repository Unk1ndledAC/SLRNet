import os, torch, tqdm
from dehazenet import DehazeNet, loss_dehaze, HazyDataset
from torch.utils.data import DataLoader

lr_dehaze = 4e-3
batch_size = 32
epochs = 10
crop_lr = 256
crop_length = 5000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_dehaze(train_pairs, model_path='weights'):
    train_set = HazyDataset(train_pairs, crop=crop_lr, length=crop_length)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    net = DehazeNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr_dehaze)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    os.makedirs(model_path, exist_ok=True)

    for epoch in range(1, epochs+1):
        net.train()
        running = {'loss':0, 'psnr':0, 'ssim':0, 'cnt':0}
        pbar = tqdm.tqdm(train_loader, ncols=100)
        for I, J in pbar:
            I, J = I.to(device), J.to(device)
            J_hat = net(I)
            loss = loss_dehaze(J_hat, J)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                running['loss'] += loss.item() * I.size(0)
                running['cnt']  += I.size(0)
                
            pbar.set_description(
                f'[Epoch {epoch}] L={running["loss"]/running["cnt"]:.4f} '
            )
        sched.step()
        
        torch.save(net.state_dict(), f'{model_path}/last.pth')
    torch.save(net.state_dict(), f'{model_path}/best.pth')
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    from dehazenet.utils import get_pairs
    train_pairs = get_pairs('./data/RESIDE_6K/train')
    train_dehaze(train_pairs)



