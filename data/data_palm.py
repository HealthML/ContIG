import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from os.path import join
from glob import glob

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import toml


torch.multiprocessing.set_sharing_strategy("file_system")

BASE = toml.load(join(os.path.dirname(os.path.realpath(__file__)), "../paths.toml"))[
    "PALM_PATH"
]


def get_tfms(size=256):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfms = transforms.Compose(
        [
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    mask_tfms = transforms.Compose(
        [
            transforms.Resize(
                size=(size, size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ]
    )

    return tfms, mask_tfms


def get_palm_loaders(
    size, batch_size=64, num_workers=8, joint_mask=True, train_pct=0.6, val_pct=0.2
):
    """get dataloaders for APTOS dataset, and also return number of labels"""
    loaders = []
    tfms, mask_tfms = get_tfms(size=size)
    for split in ["train", "valid", "test"]:
        D = PALM(
            split=split,
            tfms=tfms,
            mask_tfms=mask_tfms,
            joint_mask=joint_mask,
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


class PALM(Dataset):
    def __init__(
        self,
        split="train",
        tfms=None,
        mask_tfms=None,
        joint_mask=True,
        split_seed=42,
        train_pct=0.6,
        val_pct=0.2,
    ):
        img_subdir = "PALM-Training400"
        disk_subdir = "Disc_Masks"
        atrophy_subdir = "Lesion_Masks/Atrophy/"
        detachment_subdir = "Lesion_Masks/Detachment/"

        img_paths = glob(join(BASE, img_subdir, "*.jpg"))
        ids = [x.split("/")[-1].split(".")[0] for x in img_paths]

        disks = [
            x if os.path.isfile(x) else ""
            for x in [join(BASE, disk_subdir, f"{id}.bmp") for id in ids]
        ]
        atrophies = [
            x if os.path.isfile(x) else ""
            for x in [join(BASE, atrophy_subdir, f"{id}.bmp") for id in ids]
        ]
        detachments = [
            x if os.path.isfile(x) else ""
            for x in [join(BASE, detachment_subdir, f"{id}.bmp") for id in ids]
        ]
        classes = [1 * (x[0] == "H") for x in ids]

        df = pd.DataFrame(
            {
                "img_path": img_paths,
                "disk": disks,
                "atrophy": atrophies,
                "detachment": detachments,
                "classes": classes,
            }
        )

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=1 - train_pct, random_state=split_seed
        )
        [(train_inds, val_test_inds)] = sss.split(df.index, df.classes)
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - val_pct / (1 - train_pct),
            random_state=split_seed + 1,
        )
        [(val_inds, test_inds)] = sss.split(
            df.loc[val_test_inds], df.loc[val_test_inds].classes
        )
        val_inds = val_test_inds[val_inds]
        test_inds = val_test_inds[test_inds]

        if split == "train":
            inds = train_inds
        elif split in ["val", "valid"]:
            inds = val_inds
        elif split == "test":
            inds = test_inds
        else:
            raise ValueError(split)

        self.df = df.loc[inds]

        self.tfms = tfms
        self.joint_mask = joint_mask
        self.mask_tfms = mask_tfms

    def __len__(self):
        return len(self.df)

    def load_default(self, p, def_size=(256, 256)):
        if os.path.isfile(p):
            img = Image.open(p)
        else:
            img = Image.new("L", size=def_size, color=255)
        return img

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        inst = self.df.iloc[idx]
        img = Image.open(inst.img_path)
        disk = Image.open(inst.disk)
        atrophy = self.load_default(inst.atrophy, def_size=img.size)
        if self.tfms:
            img = self.tfms(img)
        if self.mask_tfms:
            disk = 1 - self.mask_tfms(disk)
            atrophy = 1 - self.mask_tfms(atrophy)
        if self.joint_mask:
            mask = torch.cat([disk, atrophy])
            return img, mask
        else:
            return img, disk, atrophy
