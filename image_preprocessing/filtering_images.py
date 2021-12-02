"""basic quality control for retinal fundus images

`drop_qc_paths` filters out the top_p*100% brightest and bot_p*100% darkest images
"""
from glob import glob
from os.path import join

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# TODO: fill in your paths
BASE_IMG = "PATH/TO/IMAGES/"
LEFT = join(BASE_IMG, "left/512_left/processed")
RIGHT = join(BASE_IMG, "right/512_right/processed")
IMG_EXT = ".jpg"


def drop_qc_paths(
    fn=join(BASE_IMG, "{eye}", "qc_paths_{eye}.txt"),
    top_p=0.005,
    bot_p=0.005,
):
    for base, eye in [(LEFT, "left"), (RIGHT, "right")]:
        paths = get_paths(base, subset=None)
        B = compute_brightness(paths)
        ind = np.argsort(B)
        N = len(B)
        B = B[ind]
        paths = np.array(paths)[ind]
        paths = paths[int(bot_p * N) : int((1 - top_p) * N)]
        pd.DataFrame([p.split("/")[-1] for p in paths]).to_csv(
            fn.format(eye=eye), index=None, header=None
        )


def get_paths(base, subset=None):
    return glob(join(base, "*.jpg"))[:subset]


def compute_brightness(paths):
    brightnesses = []
    for p in tqdm(paths):
        brightness = np.array(Image.open(p)).mean()
        brightnesses.append(brightness)
    return np.array(brightnesses)
