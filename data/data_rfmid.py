import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from os.path import join
from functools import partial
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
import toml


torch.multiprocessing.set_sharing_strategy("file_system")

BASE = toml.load(join(os.path.dirname(os.path.realpath(__file__)), "../paths.toml"))[
    "RFMID_PATH"
]


def get_tfms(size=256, augmentation=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if augmentation:
        tfms = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        tfms = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return tfms


def get_rfmid_loaders(
    size,
    batch_size=64,
    num_workers=8,
):
    """get dataloaders for RFMiD dataset, and also return number of labels"""
    loaders = []
    for split in ["train", "valid", "test"]:
        tfms = get_tfms(size=size, augmentation=split == "train")
        D = RFMiD(
            split=split, tfms=tfms, drop_disease_risk_hr_odpm=True, use_cropped=True
        )
        loader = DataLoader(
            D,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)

    return loaders


class RFMiD(Dataset):
    def __init__(
        self, split="train", tfms=None, drop_disease_risk_hr_odpm=True, use_cropped=True
    ):
        if split == "train":
            subdir = "Training_Set"
            label_fn = "RFMiD_Training_Labels.csv"
            img_subdir = "Training"
        elif split == "val" or split == "valid":
            subdir = "Evaluation_Set"
            label_fn = "RFMiD_Validation_Labels.csv"
            img_subdir = "Validation"
        elif split == "test":
            subdir = "Test_Set"
            label_fn = "RFMiD_Testing_Labels.csv"
            img_subdir = "Test"
        else:
            raise ValueError(f"split {split} not valid")
        if use_cropped:
            img_subdir = img_subdir + "_cropped"
        label_pth = join(BASE, subdir, subdir, label_fn)
        self.labels = pd.read_csv(label_pth, index_col=0)
        self.ext = ".png"
        if drop_disease_risk_hr_odpm:
            self.labels = self.labels.drop("Disease_Risk", 1)
            self.labels = self.labels.drop("HR", 1)
            self.labels = self.labels.drop("ODPM", 1)
            self.ext = ".jpg"
        self.img_dir = join(BASE, subdir, subdir, img_subdir)
        self.tfms = tfms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        inst = self.labels.iloc[idx]
        labels = inst.values
        id = inst.name
        p = join(self.img_dir, str(id) + self.ext)
        img = Image.open(p)
        if self.tfms:
            img = self.tfms(img)
        return img, np.array(labels, dtype=np.float)


#### data standardization utils
def crop_resize_all(split, dst, size=512, buffer=10, n_jobs=10):
    """prepare RFMiD images by center-crop-padding"""
    os.makedirs(dst, exist_ok=True)
    tfms = transforms.Compose(
        [
            partial(center_crop_pad, buffer=buffer),
            transforms.Resize(size),
        ]
    )
    D = RFMiD(split=split, tfms=tfms)
    Parallel(n_jobs=n_jobs)(
        delayed(lambda i: D[i][0].save(join(dst, f"{i+1}.jpg")))(i)
        for i in tqdm(range(len(D)))
    )


def center_crop_pad(img, buffer=0, min_mean=10):
    """dynamically center crop image, cropping away black space left and right"""
    g = np.array(img).mean(-1)
    h, w = g.shape
    zeros = g.mean(0)
    zero_inds = np.where(zeros < min_mean)[0]
    lo, hi = zero_inds[zero_inds < w // 2].max(), zero_inds[zero_inds > w // 2].min()
    return expand2square(img.crop((lo - buffer, 0, hi + buffer, h)))


def expand2square(pil_img, background_color=0):
    """from https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/"""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
