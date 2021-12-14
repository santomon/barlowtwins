import torchvision.transforms as transforms
import torch
from PIL import Image
import os
import torch.utils.data as data


class CustomSartorius(data.Dataset):

    def __init__(self, dpath, transform=None):
        self.dpath = dpath
        self.transform = transform
        self.img_list = os.listdir(dpath)
        self.img_list.sort()

    def __getitem__(self, index):
        return self.transform(Image.open(os.path.join(self.dpath, self.img_list[index])).convert("RGB"))

    def __len__(self):
        return len(self.img_list)
