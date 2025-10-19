import cv2, random
from torch.utils.data import Dataset
from torchvision import transforms

class HazyDataset(Dataset):
    def __init__(self, pairs, crop=64, length=None):
        self.crop = crop
        self.trans = transforms.ToTensor()
        self.data = []
        for gp, hp in pairs:
            j = cv2.imread(gp)[:, :, ::-1].copy()
            i = cv2.imread(hp)[:, :, ::-1].copy()
            if i.shape != j.shape:
                i = cv2.resize(i, (j.shape[1], j.shape[0]), interpolation=cv2.INTER_CUBIC)
            self.data.append((self.trans(i), self.trans(j)))

        if length is not None and length > len(self.data):
            repeat = (length // len(self.data)) + 1
            self.data = (self.data * repeat)[:length]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        I, J = self.data[idx]
        if self.crop > 0:
            _, H, W = I.shape
            top = random.randint(0, H - self.crop)
            left = random.randint(0, W - self.crop)
            I = I[:, top:top + self.crop, left:left + self.crop]
            J = J[:, top:top + self.crop, left:left + self.crop]
        return I, J