import argparse
import os
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from os.path import join
from glob import glob
from joblib import Parallel, delayed

from tqdm import tqdm

import PIL.Image
import cv2

cv2.setNumThreads(0)
from skimage.draw import circle_perimeter


warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inp", type=str, help="path to input directory")
    parser.add_argument("out", type=str, help="path for the resized images")
    parser.add_argument("--ext", type=str, default=".jpg", help="output extension")
    parser.add_argument(
        "--num_workers", type=int, default=10, help="number of parallel workers"
    )
    parser.add_argument("--size", type=int, default=672, help="resize to which size")
    args = parser.parse_args()

    resize_all_imgs(
        args.inp, args.out, args.size, extension=args.ext, n_jobs=args.num_workers
    )


def detect_circle(p, buf=5):
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


def resize_all_imgs(input_dir, output_dir, max_size, extension=".png", n_jobs=10):
    img_files = glob(join(input_dir, "*"))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(join(output_dir, "failed"), exist_ok=True)
        os.makedirs(join(output_dir, "processed"), exist_ok=True)
    Parallel(n_jobs=n_jobs)(
        delayed(resize_one)(fn, max_size, output_dir, extension=extension)
        for fn in tqdm(img_files)
    )


def resize_one(fn, max_size, output_dir, buffer=5, extension=".jpg"):
    try:
        img = PIL.Image.open(fn)
        box = detect_circle(fn, buf=buffer)
        fnn = fn.split("/")[-1].split(".")[0] + extension
        if box is None:
            img.save(join(output_dir, "failed", fnn))
        else:
            img.crop(box).resize((max_size, max_size)).save(
                join(output_dir, "processed", fnn)
            )
    except Exception as e:
        print(e)
        print("cannot handle file", fn)


if __name__ == "__main__":
    main()
