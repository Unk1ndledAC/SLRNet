import cv2, random, tqdm
from torch.utils.data import Dataset
from torchvision import transforms

class HazyDataset(Dataset):
    def __init__(self, pairs, crop=256, length=None, preload=False):
        self.crop   = crop
        self.pairs  = pairs
        self.trans  = transforms.ToTensor()
        self.preload = preload
        if length is not None and length > len(pairs):
            repeat = (length // len(pairs)) + 1
            self.pairs = (self.pairs * repeat)[:length]

        if preload:
            self.data = []
            for gp, hp in tqdm.tqdm(self.pairs, desc='Pre-loading'):
                j = cv2.imread(gp)[:,:,::-1].copy()
                i = cv2.imread(hp)[:,:,::-1].copy()
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
            j = cv2.imread(gp)[:,:,::-1].copy()
            i = cv2.imread(hp)[:,:,::-1].copy()
            if i.shape != j.shape:
                i = cv2.resize(i, (j.shape[1], j.shape[0]))
            I, J = self.trans(i), self.trans(j)

        if self.crop > 0:
            _, H, W = I.shape
            top  = random.randint(0, H - self.crop)
            left = random.randint(0, W - self.crop)
            I = I[:, top:top+self.crop, left:left+self.crop]
            J = J[:, top:top+self.crop, left:left+self.crop]
        return I, J
        

