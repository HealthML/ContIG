import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from os.path import join
from functools import partial

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
    "APTOS_PATH"
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


def get_aptos_loaders(
    size,
    batch_size=64,
    num_workers=8,
    multilabel=True,
    train_pct=0.6,
    val_pct=0.2,
):
    """get dataloaders for APTOS dataset, and also return number of labels"""
    loaders = []
    for split in ["train", "valid", "test"]:
        tfms = get_tfms(size=size, augmentation=split == "train")
        D = APTOS(
            split=split,
            tfms=tfms,
            multilabel=multilabel,
            train_pct=train_pct,
            val_pct=val_pct,
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


class APTOS(Dataset):
    def __init__(
        self,
        split="train",
        tfms=None,
        split_seed=42,
        train_pct=0.6,
        val_pct=0.2,
        use_cropped=True,
        multilabel=True,
    ):
        label_fn = "train.csv"
        img_subdir = "train_images"
        if use_cropped:
            img_subdir = img_subdir + "_cropped"

        label_pth = join(BASE, label_fn)
        self.labels = pd.read_csv(label_pth)

        # data split
        rng = np.random.RandomState(split_seed)
        N = len(self.labels)
        perm = rng.permutation(N)
        m = int(N * train_pct)
        mv = int(N * (train_pct + val_pct))
        if split == "train":
            self.labels = self.labels.iloc[perm[:m]]
        elif split in ["val", "valid"]:
            self.labels = self.labels.iloc[perm[m:mv]]
        elif split == "test":
            self.labels = self.labels.iloc[perm[mv:]]
        else:
            raise ValueError(f"split {split} not a valid option")
        self.ext = ".png"
        self.img_dir = join(BASE, img_subdir)
        self.tfms = tfms
        self.multilabel = multilabel

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        inst = self.labels.iloc[idx]
        label = inst.diagnosis
        if self.multilabel:
            if label == 0:
                label = [1, 0, 0, 0, 0]
            elif label == 1:
                label = [1, 1, 0, 0, 0]
            elif label == 2:
                label = [1, 1, 1, 0, 0]
            elif label == 3:
                label = [1, 1, 1, 1, 0]
            elif label == 4:
                label = [1, 1, 1, 1, 1]
            label = np.array(label)
        id = inst.id_code
        p = join(self.img_dir, str(id) + self.ext)
        img = Image.open(p)
        if self.tfms:
            img = self.tfms(img)
        return img, label.astype(np.float)


#### data standardization utils
def crop_resize_all(split, dst, size=512, buffer=10, n_jobs=10):
    """prepare APTOS images by disc-cropping or center-crop-padding"""
    os.makedirs(dst, exist_ok=True)
    tfms = transforms.Compose(
        [
            partial(center_crop_pad, buffer=buffer),
            transforms.Resize(size),
        ]
    )
    label_pth = join(BASE, "test.csv" if split == "test" else "train.csv")
    img_subdir = "test_images" if split == "test" else "train_images"
    paths = [
        join(BASE, img_subdir, f"{id}.png") for id in pd.read_csv(label_pth).id_code
    ]
    out_paths = [join(dst, p.split("/")[-1]) for p in paths]

    def process(p):
        box = detect_circle(p)
        img = Image.open(p)
        if box is not None:
            img = img.crop(box)
        else:
            img = center_crop_pad(img)
        return img.resize((size, size))

    for i in tqdm(range(len(paths))):
        img = process(paths[i])
        img.save(out_paths[i])


def center_crop_pad(img, buffer=0, min_mean=10):
    """dynamically center crop image, cropping away black space left and right"""
    g = np.array(img).mean(-1)
    h, w = g.shape
    zeros = g.mean(0)
    zero_inds = np.where(zeros < min_mean)[0]
    if len(zero_inds) == 0 or zero_inds.min() > w // 2 or zero_inds.max() < w // 2:
        return expand2square(img)
    lo, hi = zero_inds[zero_inds < w // 2].max(), zero_inds[zero_inds > w // 2].min()
    return expand2square(img.crop((lo - buffer, 0, hi + buffer, h)))


def detect_circle(p, buf=5):
    # only for preprocessing, so no need to install otherwise
    import cv2
    from skimage.draw import circle_perimeter

    img = cv2.imread(p)
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 25)
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=50,
        minRadius=min(h, w) // 4,
    )
    if circles is None:
        return
    else:
        C = circles[0, 0].round().astype(int)
        cc, rr = circle_perimeter(C[0], C[1], C[2])
        return [cc.min() - buf, rr.min() - buf, cc.max() + buf, rr.max() + buf]


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
