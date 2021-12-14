import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os
from PIL import Image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='directory for data')
args = parser.parse_args()


class SartoriusNormDataSet(data.Dataset):
    def __init__(self, dpath):
        self.dpath = dpath
        self.img_list = os.listdir(dpath)

    def __getitem__(self, index):
        return torch.as_tensor(Image.open(os.path.join(self.dpath,self.img_list[index])))

    def __len__(self):
        return len(self.dpath)


if __name__ == '__main__':
    dataset = SartoriusNormDataSet(args.data_dir)
    dataloader = data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=1)
    data = next(iter(dataloader))
    mean = data.mean([1, 2])
    std = data.std([1, 2])
    print(mean, std)
