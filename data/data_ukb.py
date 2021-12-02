import os

from lightly.transforms import GaussianBlur

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from collections import defaultdict
from glob import glob
from os.path import join
from typing import List

import h5py
import lightly.data as ldata
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from pysnptools.snpreader import Bed
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
import toml

torch.multiprocessing.set_sharing_strategy("file_system")

DEBUG = False

# Consts ---------------------------
config = toml.load(join(os.path.dirname(os.path.realpath(__file__)), "../paths.toml"))
BASE_IMG = config["BASE_IMG"]
LEFT = join(BASE_IMG, config["LEFT_SUBDIR"])
RIGHT = join(BASE_IMG, config["RIGHT_SUBDIR"])
IMG_EXT = ".jpg"

PHENO = config["UKB_PHENO_FILE"]

PATH_TO_COV = config["PATH_TO_COV"]

BASE_GEN = config["BASE_GEN"]

BASE_BURDEN = config["BASE_BURDEN"]

BASE_PGS = config["BASE_PGS"]
BLOOD_BIOMARKERS = config["BLOOD_BIOMARKERS"]
ANCESTRY = config["ANCESTRY"]

COVAR = [
    "eid",
    "31-0.0",  # sex
    "21022-0.0",  # age
    "4079-0.0",
    "4079-0.1",  # DBP
    "4080-0.0",
    "4080-0.1",  # SBP
    "20116-0.0",  # smoking; 2 == current; -3==prefer not to answer
    "21001-0.0",  # bmi
]
GENET_PCS = [f"22009-0.{i}" for i in range(1, 41)]

COVAR_NAMES = [
    "sex",
    "age",
    "BMI",
    "smoking",
    "SBP",
    "DBP",
] + [f"genet_pc_{i}" for i in range(1, 41)]

# Classes --------------------------------


class UKBRetina(Dataset):
    def __init__(
        self,
        eye="left",
        iid_selection=None,
        tfms=None,
        subset=None,
        return_iid=False,
        normalize_features=True,
        img_extension=IMG_EXT,
        cov_fillna="mean",
        include_biomarkers=False,
        biomarkers_filter_nan_cols=0.2,
        biomarkers_filter_nan_rows=1.0,
    ):
        self.return_iid = return_iid
        self.img_extension = img_extension
        self.tfms = tfms
        self.eye = eye
        if eye == "left":
            self.path = LEFT
        elif eye == "right":
            self.path = RIGHT
        else:
            raise ValueError()
        if iid_selection is None:
            iid_selection = get_indiv()
        self._process_ids(iid_select=iid_selection)
        self._load_covs(
            normalize_features,
            cov_fillna=cov_fillna,
            include_biomarkers=include_biomarkers,
            biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
            biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
        )
        self.subset = subset
        if subset:
            self.paths = self.paths[:subset]
            self.iids = self.iids[:subset]

    def __len__(self):
        return len(self.iids)

    def __getitem__(self, idx):
        img = self._load_img_item(idx)
        iid = self.iids[idx]
        cov = torch.from_numpy(self.cov_np[idx]).float()
        if self.return_iid:
            return img, cov, iid
        else:
            return img, cov

    def _load_img_item(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        p = self.paths[idx]
        img = Image.open(p)
        if self.tfms:
            img = self.tfms(img)
        return img

    def _load_covs(
        self,
        normalize_features,
        include_biomarkers,
        biomarkers_filter_nan_cols,
        biomarkers_filter_nan_rows,
        cov_fillna,
    ):
        cov = pd.read_csv(PATH_TO_COV, index_col=0)
        self.cov = cov.loc[self.iids]
        cols = self.cov.columns.tolist()
        df_sex_smoking = self.cov[["sex", "smoking"]].copy()
        self.cov = self.cov[self.cov.columns.difference(["sex", "smoking"])]
        if include_biomarkers:
            biomarkers = get_biomarker_data(
                filter_nan_cols=biomarkers_filter_nan_cols,
                filter_nan_rows=biomarkers_filter_nan_rows,
                iids=self.iids,
            )
            self.cov = pd.merge(
                self.cov, biomarkers, left_index=True, right_index=True, how="outer"
            )
        if cov_fillna == "mean" or cov_fillna is True:
            self.cov = self.cov.fillna(self.cov.mean())
            df_sex_smoking = df_sex_smoking.fillna(df_sex_smoking.median())
        elif cov_fillna == "median":
            self.cov = self.cov.fillna(self.cov.median())
            df_sex_smoking = df_sex_smoking.fillna(df_sex_smoking.median())
        elif (cov_fillna != False) and (cov_fillna is not None):
            raise NotImplementedError(
                f"covariate/biomarker fillna method {cov_fillna} is not yet implemented"
            )

        if normalize_features:
            scaler = StandardScaler()
            self.cov[:] = scaler.fit_transform(self.cov.values)
        self.cov = pd.concat([self.cov, df_sex_smoking], axis=1)
        self.cov = self.cov[cols]
        self.cov_np = self.cov.to_numpy()
        self.cov_columns = list(self.cov.columns)

    def _process_ids(self, iid_select=None):
        qc_fns = pd.read_csv(
            join(BASE_IMG, self.eye, f"qc_paths_{self.eye}.txt"), header=None
        ).values.flatten()
        qc_paths = np.array([join(self.path, fn) for fn in qc_fns])
        iids = np.array([int(fn.split("_")[0]) for fn in qc_fns])

        if iid_select is not None:
            iid_select = set(iid_select)
            self.paths = np.array(
                [p for iid, p in zip(iids, qc_paths) if iid in iid_select]
            )
            self.iids = np.array([iid for iid in iids if iid in iid_select])
        else:
            self.iids = iids
            self.paths = qc_paths


class UKBRetinaGen(UKBRetina):
    gen = None
    gen_lookup = None
    feature_names = None

    def __init__(
        self,
        chromos,
        rsids=None,
        sid_slice=slice(0, None, 100),
        iid_selection=None,
        eye="left",
        tfms=None,
        subset=None,
        fillna=True,
        return_iid=False,
        normalize_features=True,
        cov_fillna="mean",
        include_biomarkers=False,
        biomarkers_filter_nan_cols=0.2,
        biomarkers_filter_nan_rows=1.0,
    ):
        super().__init__(
            eye=eye,
            iid_selection=iid_selection,
            tfms=tfms,
            subset=subset,
            return_iid=return_iid,
            normalize_features=normalize_features,
            cov_fillna=cov_fillna,
            include_biomarkers=include_biomarkers,
            biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
            biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
        )
        if UKBRetinaGen.gen is None:
            gen = get_gen_data(
                chromos=chromos,
                rsids=rsids,
                sid_slice=sid_slice,
            )

            # imputing the missing SNP values (NaNs) with column-wise mode
            if fillna:
                for column in gen:
                    if gen[column].isnull().any():
                        gen[column].fillna(gen[column].mode()[0], inplace=True)
            UKBRetinaGen.gen = gen.to_numpy()
            UKBRetinaGen.gen_lookup = dict(
                (iid, idx) for idx, iid in enumerate(gen.index)
            )
            UKBRetinaGen.feature_names = list(gen.columns)

        gen_iids = np.array(list(UKBRetinaGen.gen_lookup.keys()))
        inter_iids = set(np.intersect1d(self.iids, gen_iids))

        inds = np.array([i for i, iid in enumerate(self.iids) if iid in inter_iids])
        self.paths = self.paths[inds]
        self.iids = self.iids[inds]
        self.cov = self.cov.iloc[inds]
        self.cov_np = self.cov_np[inds]

    def __getitem__(self, idx):
        img = self._load_img_item(idx)
        iid = self.iids[idx]
        cov = torch.from_numpy(self.cov_np[idx]).float()
        gen_idx = self.gen_lookup[iid]
        gen = torch.from_numpy(UKBRetinaGen.gen[gen_idx]).float()
        if self.return_iid:
            return img, cov, gen, iid
        else:
            return img, cov, gen


class UKBRetinaBurden(UKBRetina):
    burdens = None
    iid_lookup = None
    feature_names = None

    def __init__(
        self,
        filter_zeros=0.01,
        eye="left",
        iid_selection=None,
        tfms=None,
        subset=None,
        return_iid=False,
        normalize_features=True,
        cov_fillna="mean",
        include_biomarkers=False,
        biomarkers_filter_nan_cols=0.2,
        biomarkers_filter_nan_rows=1.0,
    ):
        super().__init__(
            eye=eye,
            iid_selection=iid_selection,
            tfms=tfms,
            subset=subset,
            return_iid=return_iid,
            normalize_features=normalize_features,
            cov_fillna=cov_fillna,
            include_biomarkers=include_biomarkers,
            biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
            biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
        )
        if UKBRetinaBurden.burdens is None:
            burdens = get_burden_data(filter_zeros=filter_zeros)
            burden_iids = burdens.index.to_numpy()
            UKBRetinaBurden.burdens = burdens.to_numpy()
            UKBRetinaBurden.iid_lookup = dict(
                (iid, idx) for idx, iid in enumerate(burden_iids)
            )
            UKBRetinaBurden.feature_names = list(burdens.columns)

        burden_iids = np.array(list(UKBRetinaBurden.iid_lookup.keys()))
        inter_iids = set(np.intersect1d(self.iids, burden_iids))

        inds = np.array([i for i, iid in enumerate(self.iids) if iid in inter_iids])
        self.paths = self.paths[inds]
        self.iids = self.iids[inds]
        self.cov = self.cov.iloc[inds]
        self.cov_np = self.cov_np[inds]

    def __getitem__(self, idx):
        img = self._load_img_item(idx)
        cov = torch.from_numpy(self.cov_np[idx]).float()
        iid = self.iids[idx]
        burden_idx = self.iid_lookup[iid]
        burdens = torch.from_numpy(UKBRetinaBurden.burdens[burden_idx]).float()
        if self.return_iid:
            return img, cov, burdens, iid
        else:
            return img, cov, burdens


class UKBRetinaPGS(UKBRetina):
    pgs = None
    iid_lookup = None

    def __init__(
        self,
        normalize_pgs=True,
        eye="left",
        iid_selection=None,
        tfms=None,
        subset=None,
        return_iid=False,
        normalize_features=True,
        cov_fillna="mean",
        include_biomarkers=False,
        biomarkers_filter_nan_cols=0.2,
        biomarkers_filter_nan_rows=1.0,
    ):
        super().__init__(
            eye=eye,
            iid_selection=iid_selection,
            tfms=tfms,
            subset=subset,
            return_iid=return_iid,
            normalize_features=normalize_features,
            cov_fillna=cov_fillna,
            include_biomarkers=include_biomarkers,
            biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
            biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
        )
        if UKBRetinaPGS.pgs is None:
            pgs = get_pgs_data(normalize=normalize_pgs)
            pgs_iids = pgs.index.to_numpy()
            UKBRetinaPGS.pgs = pgs.to_numpy()
            UKBRetinaPGS.iid_lookup = dict(
                (iid, idx) for idx, iid in enumerate(pgs_iids)
            )

        pgs_iids = np.array(list(UKBRetinaPGS.iid_lookup.keys()))
        inter_iids = set(np.intersect1d(self.iids, pgs_iids))

        inds = np.array([i for i, iid in enumerate(self.iids) if iid in inter_iids])
        self.paths = self.paths[inds]
        self.iids = self.iids[inds]
        self.cov = self.cov.iloc[inds]
        self.cov_np = self.cov_np[inds]

    def __getitem__(self, idx):
        img = self._load_img_item(idx)
        cov = torch.from_numpy(self.cov_np[idx]).float()
        iid = self.iids[idx]
        pgs_idx = self.iid_lookup[iid]
        pgs = torch.from_numpy(UKBRetinaPGS.pgs[pgs_idx]).float()
        if self.return_iid:
            return img, cov, pgs, iid
        else:
            return img, cov, pgs


class UKBRetinaMultimodal(UKBRetina):
    gen = None
    gen_lookup = None
    gen_feature_names = None

    pgs = None
    pgs_lookup = None

    burdens = None
    burdens_lookup = None
    burdens_feature_names = None

    def __init__(
        self,
        # gen (raw SNPs):
        gen_chromos=[i for i in range(1, 23)],
        gen_rsids=None,
        gen_sid_slice=slice(0, None, 100),
        gen_fillna=True,
        # inner (=intersection, no missings) or outer (=union, with missings)
        aggregate_modalities="inner",
        modalities=["raw_snps", "risk_scores", "burden_scores"],
        # pgs:
        normalize_pgs=True,
        # burdens;
        filter_burdens=0.01,
        # general:
        eye="left",
        iid_selection=None,
        tfms=None,
        subset=None,
        return_iid=False,
        normalize_features=True,
        cov_fillna="mean",
        include_biomarkers=False,
        biomarkers_filter_nan_cols=0.2,
        biomarkers_filter_nan_rows=1.0,
    ):
        super().__init__(
            eye=eye,
            iid_selection=iid_selection,
            tfms=tfms,
            subset=subset,
            return_iid=return_iid,
            normalize_features=normalize_features,
            cov_fillna=cov_fillna,
            include_biomarkers=include_biomarkers,
            biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
            biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
        )
        self.modalities = modalities
        gen_iids, burden_iids, pgs_iids = np.array([]), np.array([]), np.array([])
        if "raw_snps" in modalities and UKBRetinaMultimodal.gen is None:
            print("loading raw genetic data...")
            gen = get_gen_data(
                chromos=gen_chromos,
                rsids=gen_rsids,
                sid_slice=gen_sid_slice,
            )
            # imputing the missing SNP values (NaNs) with column-wise mode
            if gen_fillna:
                for column in gen:
                    if gen[column].isnull().any():
                        gen[column].fillna(gen[column].mode()[0], inplace=True)
            UKBRetinaMultimodal.gen = gen.to_numpy()
            UKBRetinaMultimodal.gen_lookup = defaultdict(
                lambda: None, [(iid, idx) for idx, iid in enumerate(gen.index)]
            )
            UKBRetinaMultimodal.gen_feature_names = list(gen.columns)
        if UKBRetinaMultimodal.gen_lookup is not None:
            gen_iids = np.array(list(UKBRetinaMultimodal.gen_lookup.keys()))
        if "risk_scores" in modalities and UKBRetinaMultimodal.pgs is None:
            print("loading polygenic risk score data...")
            pgs = get_pgs_data(normalize=normalize_pgs)
            pgs_iids = pgs.index.to_numpy()
            UKBRetinaMultimodal.pgs = pgs.to_numpy()
            UKBRetinaMultimodal.pgs_lookup = defaultdict(
                lambda: None, [(iid, idx) for idx, iid in enumerate(pgs_iids)]
            )
        if UKBRetinaMultimodal.pgs_lookup is not None:
            pgs_iids = np.array(list(UKBRetinaMultimodal.pgs_lookup.keys()))
        if "burden_scores" in modalities and UKBRetinaMultimodal.burdens is None:
            print("loading burden score data...")
            burdens = get_burden_data(filter_zeros=filter_burdens)
            burden_iids = burdens.index.to_numpy()
            UKBRetinaMultimodal.burdens = burdens.to_numpy()
            UKBRetinaMultimodal.burdens_lookup = defaultdict(
                lambda: None, [(iid, idx) for idx, iid in enumerate(burden_iids)]
            )
            UKBRetinaMultimodal.burdens_feature_names = list(burdens.columns)
        if UKBRetinaMultimodal.burdens_lookup is not None:
            burden_iids = np.array(list(UKBRetinaMultimodal.burdens_lookup.keys()))

        if (
            aggregate_modalities == "inner"
        ):  # make sure that all genetic modalities are available
            if gen_iids.size == 0 and pgs_iids.size != 0 and burden_iids.size != 0:
                selected_iids = set(
                    np.intersect1d(
                        self.iids,
                        np.intersect1d(burden_iids, pgs_iids),
                    )
                )
            elif gen_iids.size != 0 and pgs_iids.size == 0 and burden_iids.size != 0:
                selected_iids = set(
                    np.intersect1d(
                        self.iids,
                        np.intersect1d(burden_iids, gen_iids),
                    )
                )
            elif gen_iids.size != 0 and pgs_iids.size != 0 and burden_iids.size == 0:
                selected_iids = set(
                    np.intersect1d(
                        self.iids,
                        np.intersect1d(pgs_iids, gen_iids),
                    )
                )
            else:
                selected_iids = set(
                    np.intersect1d(
                        self.iids,
                        np.intersect1d(gen_iids, np.intersect1d(burden_iids, pgs_iids)),
                    )
                )
        elif aggregate_modalities == "outer":
            selected_iids = set(
                np.union1d(
                    self.iids,
                    np.union1d(gen_iids, np.union1d(burden_iids, pgs_iids)),
                )
            )
        else:
            raise ValueError(f"aggregation {aggregate_modalities} not known")
        inds = np.array([i for i, iid in enumerate(self.iids) if iid in selected_iids])
        self.paths = self.paths[inds]
        self.iids = self.iids[inds]
        self.cov = self.cov.iloc[inds]
        self.cov_np = self.cov_np[inds]

    def __getitem__(self, idx):
        img = self._load_img_item(idx)
        cov = torch.from_numpy(self.cov_np[idx]).float()
        iid = self.iids[idx]

        gen, pgs, burdens = torch.empty(1), torch.empty(1), torch.empty(1)
        gen_idx, pgs_idx, burdens_idx = None, None, None
        if "raw_snps" in self.modalities:
            gen_idx = self.gen_lookup[iid]
            gen = torch.from_numpy(
                np.full(self.gen.shape[1], np.nan)
                if gen_idx is None
                else self.gen[gen_idx]
            ).float()
        if "risk_scores" in self.modalities:
            pgs_idx = self.pgs_lookup[iid]
            pgs = torch.from_numpy(
                np.full(self.pgs.shape[1], np.nan)
                if pgs_idx is None
                else self.pgs[pgs_idx]
            ).float()
        if "burden_scores" in self.modalities:
            burdens_idx = self.burdens_lookup[iid]
            burdens = torch.from_numpy(
                np.full(self.burdens.shape[1], np.nan)
                if burdens_idx is None
                else self.burdens[burdens_idx]
            ).float()

        missing = torch.tensor(
            [gen_idx is None, pgs_idx is None, burdens_idx is None],
            dtype=torch.long,
        )

        if self.return_iid:
            return {
                "iid": iid,
                "img": img,
                "cov": cov,
                "gen": gen,
                "pgs": pgs,
                "burdens": burdens,
                "missing": missing,
            }
        else:
            return {
                "img": img,
                "cov": cov,
                "gen": gen,
                "pgs": pgs,
                "burdens": burdens,
                "missing": missing,
            }


# Loaders -------------------------------


def get_multimodal_pretraining_data(
    # inner (=intersection, no missings) or outer (=union, with missings)
    aggregate_modalities="inner",
    modalities=["raw_snps", "risk_scores", "burden_scores"],
    # raw genetics
    gen_chromos=[i for i in range(1, 23)],
    gen_sid_slice=slice(0, None, 100),
    # pgs
    normalize_pgs=True,
    # burdens
    burdens_zeros=0.1,  # filter burden scores by numbers of non-zeros (percentage or absolute)
    # general
    seed=42,
    num_workers=8,
    size=256,
    batch_size=32,
    train_pct=0.6,
    val_pct=0.2,
    subset=None,
    normalize_features=True,
    return_iid=False,
    tfms_settings="default",
    cov_fillna="mean",
    include_biomarkers=False,
    biomarkers_filter_nan_cols=0.2,
    biomarkers_filter_nan_rows=1.0,
):
    t_iids, v_iids, tt_iids = get_indiv_split(
        train_pct=train_pct, val_pct=val_pct, seed=seed
    )
    loaders = []
    for iids, mode in [(t_iids, "train"), (v_iids, "valid"), (tt_iids, "test")]:
        tfms = get_tfms(size=size, augmentation=mode == "train", setting=tfms_settings)
        dsets = [
            UKBRetinaMultimodal(
                aggregate_modalities=aggregate_modalities,
                modalities=modalities,
                # raw SNPs
                gen_chromos=gen_chromos,
                gen_rsids=None,
                gen_sid_slice=gen_sid_slice,
                gen_fillna=True,
                # pgs:
                normalize_pgs=normalize_pgs,
                # burdens;
                filter_burdens=burdens_zeros,
                # general:
                iid_selection=iids,
                return_iid=return_iid,
                eye=eye,
                tfms=tfms,
                subset=subset,
                normalize_features=normalize_features,
                cov_fillna=cov_fillna,
                include_biomarkers=include_biomarkers,
                biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
                biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
            )
            for eye in ["left", "right"]
        ]
        dataset = torch.utils.data.ConcatDataset(dsets)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    num_features = {
        "gen": dsets[0].gen.shape[1] if dsets[0].gen is not None else 0,
        "pgs": dsets[0].pgs.shape[1] if dsets[0].pgs is not None else 0,
        "burdens": dsets[0].burdens.shape[1] if dsets[0].burdens is not None else 0,
        "cov": dsets[0].cov.shape[1],
    }

    return loaders, num_features


def get_genetics_imaging_data(
    rsids=[],
    chromos=[i for i in range(1, 23)],
    sid_slice=slice(0, None, 100),
    burdens_zeros=None,  # filter burden scores by numbers of non-zeros (percentage or absolute)
    seed=42,
    num_workers=4,
    size=256,
    normalize_features=True,
    batch_size=32,
    train_pct=0.6,
    val_pct=0.2,
    subset=None,
    return_iid=False,
    tfms_settings="default",
    cov_fillna="mean",
    include_biomarkers=False,
    biomarkers_filter_nan_cols=0.2,
    biomarkers_filter_nan_rows=1.0,
):
    """load imaging with either raw genetic or with burden data

    for raw SNPs:
        get_genetics_imaging_data(rsids=['rs123', ...], chromos=[1, ...], sid_slice=None, ...)
    or
        get_genetics_imaging_data(rsids=None, chromos=[1, ...], sid_slice=slice(0, None, 100), ...)
    for burden data, use
        get_genetics_imaging_data(rsids=None, chromos=None, sid_slice=None, burdens_zeros=0.01, ...)

    """
    assert (
        rsids is None or sid_slice is None
    ), "specified both rsids and sid_slice; need to choose one or the other"
    assert (
        burdens_zeros is None or rsids is None
    ), "specify either burdens or snps, not both"

    t_iids, v_iids, tt_iids = get_indiv_split(
        train_pct=train_pct, val_pct=val_pct, seed=seed
    )
    loaders = []
    for iids, mode in [(t_iids, "train"), (v_iids, "valid"), (tt_iids, "test")]:
        tfms = get_tfms(size=size, augmentation=mode == "train", setting=tfms_settings)
        if burdens_zeros is None:
            dsets = [
                UKBRetinaGen(
                    iid_selection=iids,
                    chromos=chromos,
                    rsids=rsids,
                    sid_slice=sid_slice,
                    eye=eye,
                    tfms=tfms,
                    normalize_features=normalize_features,
                    subset=subset,
                    return_iid=return_iid,
                    cov_fillna=cov_fillna,
                    include_biomarkers=include_biomarkers,
                    biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
                    biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
                )
                for eye in ["left", "right"]
            ]
            gen_num_features = dsets[0].gen.shape[-1]
        else:
            dsets = [
                UKBRetinaBurden(
                    filter_zeros=burdens_zeros,
                    iid_selection=iids,
                    eye=eye,
                    tfms=tfms,
                    normalize_features=normalize_features,
                    subset=subset,
                    return_iid=return_iid,
                    cov_fillna=cov_fillna,
                    include_biomarkers=include_biomarkers,
                    biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
                    biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
                )
                for eye in ["left", "right"]
            ]
            gen_num_features = dsets[0].burdens.shape[-1]

        dataset = torch.utils.data.ConcatDataset(dsets)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    return loaders, gen_num_features


def get_imaging_pretraining_data(
    seed=42,
    num_workers=4,
    size=256,
    batch_size=50,
    train_pct=0.6,
    val_pct=0.2,
    tfms_settings="default",
):
    t_iids, v_iids, tt_iids = get_indiv_split(
        train_pct=train_pct, val_pct=val_pct, seed=seed
    )
    loaders = []

    def get_path_by_index(dataset, index):
        # filename is the path of the image relative to the dataset root
        return dataset.paths[index]

    class UKBLightlyCollateFunction(ldata.BaseCollateFunction):
        def __init__(self, transform: torchvision.transforms.Compose):
            super().__init__(transform)

        def forward(self, batch: List[tuple]):
            batch_size = len(batch)

            # list of transformed images
            transforms = [
                self.transform(batch[i % batch_size][0]).unsqueeze_(0)
                for i in range(2 * batch_size)
            ]
            # list of labels
            labels = torch.LongTensor([0 for _ in batch])
            # list of filenames
            fnames = [item[2] for item in batch]

            # tuple of transforms
            transforms = (
                torch.cat(transforms[:batch_size], 0),
                torch.cat(transforms[batch_size:], 0),
            )
            return transforms, labels, fnames

    for iids, mode in [(t_iids, "train"), (v_iids, "valid"), (tt_iids, "test")]:
        our_dsets = [
            UKBRetina(
                eye=eye,
                iid_selection=iids,
            )
            for eye in ["left", "right"]
        ]
        lightly_dsets = [
            ldata.LightlyDataset.from_torch_dataset(
                dset, index_to_filename=get_path_by_index
            )
            for dset in our_dsets
        ]
        dataset = torch.utils.data.ConcatDataset(lightly_dsets)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=UKBLightlyCollateFunction(
                transform=get_tfms(
                    size=size, augmentation=mode == "train", setting=tfms_settings
                )
            ),
        )
        loaders.append(loader)
    return loaders


def get_imaging_card_data(
    seed=42,
    num_workers=8,
    size=256,
    normalize_features=True,
    batch_size=50,
    train_pct=0.6,
    val_pct=0.2,
    subset=None,
    tfms_settings="default",
    cov_fillna="mean",
    return_iid=False,
    include_biomarkers=False,
    biomarkers_filter_nan_cols=0.2,
    biomarkers_filter_nan_rows=1.0,
):
    t_iids, v_iids, tt_iids = get_indiv_split(
        train_pct=train_pct, val_pct=val_pct, seed=seed
    )
    loaders = []
    for iids, mode in [(t_iids, "train"), (v_iids, "valid"), (tt_iids, "test")]:
        tfms = get_tfms(size=size, augmentation=mode == "train", setting=tfms_settings)
        dsets = [
            UKBRetina(
                eye=eye,
                iid_selection=iids,
                tfms=tfms,
                normalize_features=normalize_features,
                subset=subset,
                cov_fillna=cov_fillna,
                return_iid=return_iid,
                include_biomarkers=include_biomarkers,
                biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
                biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
            )
            for eye in ["left", "right"]
        ]
        dataset = torch.utils.data.ConcatDataset(dsets)
        cov_num_features = dsets[0].cov_np.shape[-1]

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    return loaders, cov_num_features


def get_pgs_imaging_data(
    normalize_pgs=True,
    seed=42,
    num_workers=4,
    size=256,
    normalize_features=True,
    batch_size=32,
    train_pct=0.6,
    val_pct=0.2,
    subset=None,
    return_iid=False,
    tfms_settings="default",
    cov_fillna="mean",
    include_biomarkers=False,
    biomarkers_filter_nan_cols=0.2,
    biomarkers_filter_nan_rows=1.0,
):
    t_iids, v_iids, tt_iids = get_indiv_split(
        train_pct=train_pct, val_pct=val_pct, seed=seed
    )
    loaders = []
    for iids, mode in [(t_iids, "train"), (v_iids, "valid"), (tt_iids, "test")]:
        tfms = get_tfms(size=size, augmentation=mode == "train", setting=tfms_settings)
        dsets = [
            UKBRetinaPGS(
                eye=eye,
                iid_selection=iids,
                tfms=tfms,
                normalize_features=normalize_features,
                normalize_pgs=normalize_pgs,
                subset=subset,
                cov_fillna=cov_fillna,
                return_iid=return_iid,
                include_biomarkers=include_biomarkers,
                biomarkers_filter_nan_cols=biomarkers_filter_nan_cols,
                biomarkers_filter_nan_rows=biomarkers_filter_nan_rows,
            )
            for eye in ["left", "right"]
        ]
        dataset = torch.utils.data.ConcatDataset(dsets)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    return loaders, dsets[0].pgs.shape[1]


def get_burden_data(filter_zeros=0):
    """load burden data and filter columns with low numbers of non-zeros

    if filter_zeros >= 1: minimum number of non-zero individuals
    if filter_zeros in (0, 1): minimum percentage of of non-zero individuals
    """
    cols = (
        pd.read_csv(join(BASE_BURDEN, "combined_burdens_colnames.txt"), header=None)
        .to_numpy()
        .flatten()
    )

    main_iids = get_indiv()
    burden_iids = (
        pd.read_csv(join(BASE_BURDEN, "combined_burdens_iid.txt"), header=None)
        .to_numpy()
        .flatten()
    )
    inds = fast_index_lookup(burden_iids, main_iids)
    inds.sort()
    iids = burden_iids[inds]
    if DEBUG:
        data = pd.DataFrame(
            data=np.random.randint(0, 10, size=(len(iids), len(cols))),
            index=iids,
            columns=cols,
        )
        return data.sort_index()

    print("loading burden data...")
    G = h5py.File(join(BASE_BURDEN, "combined_burdens.h5"))["G"][inds]

    if filter_zeros > 0:
        if filter_zeros < 1:
            filter_zeros = len(iids) * filter_zeros
        g0 = (G > 0).sum(0)
        col_ind = g0 >= filter_zeros
        print(f"selecting {col_ind.sum()} with minimum of {filter_zeros} non-zeros")
        G = G[:, col_ind]
        cols = cols[col_ind]

    data = pd.DataFrame(data=G, index=iids, columns=cols)

    return data.sort_index()


def get_biomarker_data(filter_nan_cols=0.2, filter_nan_rows=1.0, iids=None):
    """
    filter biomarkers:
        first throw out all individuals with more than 100*filter_nan_rows% NaNs
        second throw out all biomarkers with more than 100*filter_nan_cols% NaNs in the remaining data

    """
    if iids is None:
        iids = get_indiv()
    df = pd.read_csv(BLOOD_BIOMARKERS, sep="\t", index_col=0)
    inter_iids = np.intersect1d(iids, df.index)
    df = df.loc[inter_iids]
    nan_means_row = df.isna().mean(1)
    df = df.loc[nan_means_row <= filter_nan_rows]
    nan_means_col = df.isna().mean()
    df = df.loc[:, nan_means_col <= filter_nan_cols]
    return df


def get_pgs_data(normalize=True):
    available_pgs = sorted(glob(join(BASE_PGS, "*.sscore")))
    iids = get_indiv()
    for pgs_p in tqdm(available_pgs):
        pgs = pgs_p.split("/")[-1].split(".")[0]
        df = pd.read_csv(pgs_p, usecols=["IID", "SCORE1_AVG"], sep="\t", index_col=0)
        if pgs_p == available_pgs[0]:
            iids = np.intersect1d(iids, df.index)
            full_df = df.loc[iids]
            full_df.columns = [pgs]
        else:
            full_df[pgs] = df.loc[iids]
    if normalize:
        full_df = (full_df - full_df.mean()) / full_df.std()
    return full_df


def get_gen_data(chromos=[15, 19], rsids=[], sid_slice=None):
    """
    load all SNPS that have to be on one of the provided chromos
    snps that can not be found (eg if maf<0.001 or not on the microarray) will be ignored
    # ld = 0.8 throw some snps that are close to each other
    """
    ids = get_indiv()
    for chromo in tqdm(chromos):
        path_to_genetic = join(BASE_GEN, f"ukb_chr{chromo}_v2")

        bed = Bed(path_to_genetic, count_A1=False)
        ind = bed.iid_to_index([[str(i), str(i)] for i in ids])
        if sid_slice is not None:
            sid_ind = sid_slice
        else:
            sid_ind = bed.sid_to_index(np.intersect1d(rsids, bed.sid))
        labels = bed[ind, sid_ind].read().val
        df = pd.DataFrame(
            index=ids, data=labels, columns=bed.sid[sid_ind], dtype=np.float32
        )
        if chromo == chromos[0]:
            full_df = df
        else:
            full_df = pd.merge(full_df, df, left_index=True, right_index=True)
    return full_df


# Helpers -------------------------


def test_train_valid_leak(tl, vl, ttl):
    """short utility to ensure there's no data leakage"""
    D0 = tl.dataset.datasets[0]
    D1 = tl.dataset.datasets[1]
    N = len(D0)
    # tI = tl.sampler.indices
    tI = tl.sampler if isinstance(tl.sampler, torch.Tensor) else tl.sampler.indices
    vI = vl.sampler if isinstance(vl.sampler, torch.Tensor) else vl.sampler.indices
    ttI = ttl.sampler if isinstance(ttl.sampler, torch.Tensor) else ttl.sampler.indices
    train_iids = [D0.iids[i] if i < N else D1.iids[i - N] for i in tI]
    valid_iids = [D0.iids[i] if i < N else D1.iids[i - N] for i in vI]
    test_iids = [D0.iids[i] if i < N else D1.iids[i - N] for i in ttI]
    inter_valid = np.intersect1d(train_iids, valid_iids)
    inter_test = np.intersect1d(train_iids, test_iids)
    print(
        f"intersection: {len(inter_valid)} of {len(train_iids)}(train) and {len(valid_iids)}(valid)"
    )
    print(
        f"intersection: {len(inter_test)} of {len(train_iids)}(train) and {len(test_iids)}(test)"
    )


def get_indiv_split(train_pct=0.6, val_pct=0.2, seed=42):
    """train/val/test split, stratified by individuals (no data leakage)"""
    rng = np.random.RandomState(seed)
    iids = get_indiv()
    iids = rng.permutation(iids)

    m = len(iids)
    t_cut = int(train_pct * m)
    v_cut = int(val_pct * m) + t_cut
    train_iids = iids[:t_cut]
    valid_iids = iids[t_cut:v_cut]
    test_iids = iids[v_cut:]
    return train_iids, valid_iids, test_iids


def export_card():
    # encoding errors when pandas.__version__ == '1.3.x'? but works in 1.2.5
    df = pd.read_csv(PHENO, usecols=COVAR + GENET_PCS)
    iids = get_indiv()
    df["iid"] = df["eid"]
    df.index = df.iid
    df = df.loc[iids]

    df["SBP"] = df[["4080-0.0", "4080-0.1"]].mean(1)
    df["DBP"] = df[["4079-0.0", "4079-0.1"]].mean(1)

    df["age"] = df["21022-0.0"]
    df["sex"] = df["31-0.0"]
    df["smoking"] = 1 * (df["20116-0.0"] == 2)
    df["BMI"] = df["21001-0.0"]

    for i, col in enumerate(GENET_PCS):
        df[f"genet_pc_{i + 1}"] = df[col]

    df = df[
        ["sex", "age", "BMI", "smoking", "SBP", "DBP"]
        + [f"genet_pc_{i + 1}" for i in range(40)]
    ]
    df.to_csv(PATH_TO_COV, index=True)
    return df


def get_indiv(ancestry_threshold=0.99):
    all_iids = []
    for eye in ["left", "right"]:
        iids = list(
            pd.read_csv(
                join(BASE_IMG, eye, f"qc_paths_{eye}.txt"),
                header=None,
                sep="_",
                usecols=[0],
            )[0]
        )
        all_iids += iids
    iids = np.unique(all_iids)

    ancestry = pd.read_csv(ANCESTRY, sep="\t", index_col=0, usecols=["IID", "EUR"])
    anc_iids = ancestry[ancestry.EUR >= ancestry_threshold].index
    iids = np.intersect1d(iids, anc_iids)

    return iids


def get_augmented_tfms(size=224, setting="default"):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if setting == "none":
        tfms = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif setting == "default":
        tfms = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.RandomRotation(degrees=20),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif isinstance(setting, dict):
        tfms_list = [transforms.Resize(size=size)]
        if setting["rrc"] is not None:
            tfms_list.append(transforms.RandomResizedCrop(**setting["rrc"]))
        if setting["rotate"] is not None:
            tfms_list.append(transforms.RandomRotation(**setting["rotate"]))
        if setting["flip"]:
            tfms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        # TODO: this should probably be adaptive?
        if setting["blur"]:
            tfms_list.append(transforms.RandomAdjustSharpness(sharpness_factor=0.5))
        if setting["autocontrast"]:
            tfms_list.append(transforms.RandomAutocontrast(p=0.5))
        if setting["jitter"] is not None:
            tfms_list.append(transforms.ColorJitter(**setting["jitter"]))

        tfms_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        tfms = transforms.Compose(tfms_list)
    elif setting == "simclr":
        cj_prob = 0.8
        cj_bright = 0.7
        cj_contrast = 0.7
        cj_sat = 0.7
        cj_hue = 0.2
        min_scale = 0.08
        random_gray_scale = 0.2
        gaussian_blur = 0.5
        kernel_size = 0.1
        hf_prob = 0.5
        color_jitter = transforms.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)
        tfms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(min_scale, 1.0)),
                transforms.RandomHorizontalFlip(p=hf_prob),
                transforms.RandomApply([color_jitter], p=cj_prob),
                transforms.RandomGrayscale(p=random_gray_scale),
                GaussianBlur(kernel_size=kernel_size * size, prob=gaussian_blur),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return tfms


def get_tfms(size=224, augmentation=False, setting="default"):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if augmentation:
        return get_augmented_tfms(size=size, setting=setting)
    else:
        tfms = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return tfms


def fast_index_lookup(a1, a2):
    """more efficient implementation of np.where(a1 == a2[:, None])[1]

    assumes no duplicates in a2, otherwise it will only consider the last occurence of the duplicate instance
    """
    ll = defaultdict(lambda: None, [(v, i) for i, v in enumerate(a1)])
    return np.array([ll[v] for v in a2 if not ll[v] is None])
