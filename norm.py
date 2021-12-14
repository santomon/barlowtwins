import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os
from PIL import Image
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='directory for data')
args = parser.parse_args()


class SartoriusNormDataSet(data.Dataset):
    def __init__(self, dpath):
        self.dpath = dpath
        self.img_list = os.listdir(dpath)

    def __getitem__(self, index):
        return torch.as_tensor(np.array(Image.open(os.path.join(self.dpath,self.img_list[index]))), dtype=torch.float32)

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    dataset = SartoriusNormDataSet(args.data_dir)
    dataloader = data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=1)
    data = next(iter(dataloader))
    print(data.shape)
    mean = data.mean([0, 1, 2])
    std = data.std([0, 1, 2])
    print(mean, std)

#computed mean and std for Sartorius dataset
# mean:127.9788, std: 11.7118
