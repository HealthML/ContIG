import os
import shutil
import tarfile
import urllib.request as request
from contextlib import closing
from glob import glob
from io import StringIO
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

DST = "../pgs_data"
LIST = join(DST, "pgs_scores_list.txt")

# urls:
LIST_FILE = "http://ftp.ebi.ac.uk/pub/databases/spot/pgs/pgs_scores_list.txt"
BASE_URL = "http://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/"

# templates
SCORE_TEMPLATE = "{id}/ScoringFiles/{id}.txt.gz"
META_TEMPLATE = "{id}/Metadata/{id}_metadata.tar.gz"


def download_multiple_pgs(pgs_ids="all", dst=DST):
    download_ftp(LIST_FILE, LIST)
    all_ids = pd.read_csv(LIST, header=None).values.flatten()
    if pgs_ids == "all":
        dl_ids = all_ids
    else:
        dl_ids = np.intersect1d(all_ids, pgs_ids)
    for pgs_id in dl_ids:
        load_pgs(pgs_id, dst=dst)


def load_pgs(pgs_id, dst=DST):
    score_url = join(BASE_URL, SCORE_TEMPLATE.format(id=pgs_id))
    score_fn = join(dst, SCORE_TEMPLATE.format(id=pgs_id))
    download_ftp(score_url, score_fn)

    meta_url = join(BASE_URL, META_TEMPLATE.format(id=pgs_id))
    meta_fn = join(dst, META_TEMPLATE.format(id=pgs_id))
    download_ftp(meta_url, meta_fn)


def download_ftp(url, fn):
    dir = join(*fn.split("/")[:-1])
    os.makedirs(dir, exist_ok=True)
    print(f"downloading: {url} to {fn}")
    with closing(request.urlopen(url)) as r:
        with open(fn, "wb") as f:
            shutil.copyfileobj(r, f)


def list_traits(pgs_dir):
    files = sorted(glob(join(pgs_dir, "*.sscore")))
    pgs = [fn.split("/")[-1].split(".")[0] for fn in files]
    traits = []
    for i, p in tqdm(enumerate(pgs)):
        fn = join(DST, p, "Metadata", p + "_metadata.tar.gz")
        t = tarfile.open(fn, "r:gz")
        df = pd.read_csv(
            StringIO(t.extractfile(f"{p}_metadata_efo_traits.csv").read().decode())
        )
        trait = ", ".join(df["Ontology Trait Label"])

        traits.append({"pgs_id": p, "trait": trait, "index": i})

    return pd.DataFrame(traits)
