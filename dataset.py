from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement
import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    hsize = size * scale
    hx, hy = x * scale, y * scale

    crop_lr = lr[y:y + size, x:x + size].copy()
    crop_hr = hr[hy:hy + hsize, hx:hx + hsize].copy()

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.size = size
        h5f = h5py.File(path, "r")

        self.hr = [v[:] for v in h5f["HR"].values()]

        self.scale = [scale]
        self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]

        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size

        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]

        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        return len(self.hr)


class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name = dirname.split("/")[-1]
        self.scale = scale
        if "DIV" in self.name:
            self.hr = glob.glob(os.path.join("{}_HR".format(dirname), "*.png"))
            self.lr = glob.glob(os.path.join("{}_LR_bicubic".format(dirname),
                                             "X{}/*.png".format(scale)))
        elif "Set" in self.name or "BSD" in self.name or "Manga" in self.name or "Urban" in self.name:  # Set5(old)/Set14 专用结构
            # #HR路径: Set5(old)/HR/*.png
            # self.hr = glob.glob(os.path.join(dirname, "HR", "*.png"))
            # # LR路径: Set5(old)/LR_bicubic/X4/*.png
            # self.lr = glob.glob(os.path.join(dirname, "LR_bicubic", f"X{scale}", "*.png"))
            lr_path = os.path.join(dirname, "image_SRF_{}/LR/*.png".format(scale))
            hr_path = os.path.join(dirname, "image_SRF_{}/HR/*.png".format(scale))
            self.lr = glob.glob(lr_path)
            self.hr = glob.glob(hr_path)

        elif "valid" in self.name or "data/DRSRD1_2D_valid_shuffle" in self.name or "DeepRock" in self.name:
            # HR路径: dirname/valid_HR/*.png
            self.hr = glob.glob(os.path.join(dirname, "valid_HR", "*.png"))
            # LR路径: dirname/X4/X{scale}/*.png
            self.lr = glob.glob(os.path.join(dirname, "valid_LR", f"X{scale}", "*.png"))

        elif "test" in self.name:
            # HR路径: dirname/valid_HR/*.png
            self.hr = glob.glob(os.path.join(dirname, "HR", "*.png"))
            # LR路径: dirname/X4/X{scale}/*.png
            self.lr = glob.glob(os.path.join(dirname, "test_LR", f"X{scale}", "*.png"))
            # print(self.lr)

        else:
            all_files = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]

        self.hr.sort()
        self.lr.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])

        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        filename = self.hr[index].split("/")[-1]

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)
