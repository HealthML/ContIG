import os
import subprocess
from os.path import join

import numpy as np
import pandas as pd
import toml
import torch
from qmplot import manhattanplot
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_ukb import get_tfms, get_indiv_split, UKBRetina
from models.model_loading import GeneralModel

config = toml.load("paths.toml")

# need both: plink1's association computations are super slow; plink2 doesn't implement clumping
PLINK1 = config["PLINK1"]
PLINK2 = config["PLINK2"]
PRJ = config["CHECKPOINTS_BASE_PATH"]
BFILE = join(config["BASE_GEN"], "ukb_chr{chromo}_v2")

TEMPL = "gwas_results_chr{chromo}.PC{pc}.glm.linear"

# significance thresholds for clumping
P1 = 5e-8
P2 = 1e-7

SIZE = 448
COVARIATES = ["sex", "age"] + [f"genet_pc_{i}" for i in range(1, 16)]


WEIGHT_PATHS = {
    # baselines
    "barlow": join(PRJ, "barlow_r50_proj128/epoch_99-step_170399.ckpt"),
    "byol": join(PRJ, "byol_r50_proj128/epoch_99-step_170399.ckpt"),
    "nnclr": join(PRJ, "nnclr_r50_proj128/epoch_99-step_170399.ckpt"),
    "simclr": join(PRJ, "simclr_r50_proj128/epoch_99-step_170399.ckpt"),
    "simsiam": join(PRJ, "simsiam_r50_proj128/epoch_99-step_170399.ckpt"),
    # ContIG
    "rpb": join(PRJ, "cm_r50_raw_risks_burdens_outer_h1/checkpoints/last.ckpt"),
    "rpb-inner": join(PRJ, "cm_r50_raw_risks_burdens_inner_h1/last.ckpt"),
    "gen": join(PRJ, "cm_r50_raw_snps_h1/last.ckpt"),
    "pgs": join(PRJ, "cm_r50_risk_scores_gen_h1/last.ckpt"),
    "burdens": join(PRJ, "cm_r50_burden_scores_gen_h1/last.ckpt"),
}


def run_all_gwas(
    split="test",
    dev="cuda:0",
    bs=10,
    threads=20,
    main_dir="gwas_results",
    use_INT=True,
    subset=None,
):
    for key in WEIGHT_PATHS:
        print("starting model", key)
        run_transfer_gwas(
            out_dir=join(main_dir, key),
            weights_path=WEIGHT_PATHS[key],
            split=split,
            subset=subset,
            dev=dev,
            bs=bs,
            threads=threads,
            use_INT=use_INT,
        )


def compare_models(main_dir="gwas_results"):
    results = dict()
    for key in WEIGHT_PATHS:
        fn = join(main_dir, key, "final_clumps.csv")
        if os.path.isfile(fn):
            if os.path.getsize(fn) > 1:
                res = pd.read_csv(fn)
                results[key] = len(res)
            else:
                results[key] = 0
    return results


def run_transfer_gwas(
    out_dir="gwas_results",
    weights_path=WEIGHT_PATHS["rpb"],
    pheno_fn="transfer_embeddings.txt",
    cov_fn="transfer_cov.txt",
    size=SIZE,
    split="valid",
    comp=10,
    dev="cuda:0",
    threads=20,
    seed=42,
    use_INT=True,
    bs=10,
    subset=None,
):
    os.makedirs(out_dir, exist_ok=True)
    pheno_fn = join(out_dir, pheno_fn)
    cov_fn = join(out_dir, cov_fn)

    print("loading model & data...")
    model = GeneralModel(checkpoint_path=weights_path, device=dev).eval()

    tl, vl, ttl = get_gwas_data(
        size=size,
        batch_size=bs,
        return_iid=True,
        normalize_features=False,
        seed=seed,
        subset=subset,
    )
    loader = {"train": tl, "valid": vl, "test": ttl}[split]

    print(f"computing {split} embeddings")
    export_embeddings(
        loader,
        model,
        pheno_fn=pheno_fn,
        cov_fn=cov_fn,
        comp=comp,
        dev=dev,
        use_INT=use_INT,
    )

    print(f"running GWAS")
    run_plink(
        pheno_fn=pheno_fn,
        covar_fn=cov_fn,
        out=join(out_dir, "gwas_results_chr{chromo}"),
        threads=threads,
    )

    plot_gwas(
        out_fn=join(out_dir, "mhat_plot"),
        templ=join(out_dir, "gwas_results_chr{chromo}.PC{pc}.glm.linear"),
        clip_min=1e-99,
    )
    clump_results(direc=out_dir)


def plot_gwas(
    out_fn=join("gwas_results", "mhat_plot"),
    templ=join("gwas_results", "gwas_results_chr{chromo}.PC{pc}.glm.linear"),
    clip_min=1e-99,
):
    df = get_plink_results(templ=templ)
    df.P = df.P.clip(clip_min, 1)
    manhattanplot(
        data=df.dropna(how="any", axis=0),
        figname=out_fn + ".png",
        xticklabel_kws={"rotation": "vertical"},
    )
    manhattanplot(
        data=df.dropna(how="any", axis=0),
        figname=out_fn + ".pdf",
        xticklabel_kws={"rotation": "vertical"},
    )


def clump_results(
    direc,
    p1=P1,
    p2=P2,
    r2=0.1,
    kb=150,
    threads=20,
):
    merged_fn = join(direc, "gwas_merged.txt")
    merge_plink(
        templ=join(direc, TEMPL),
        out_fn=merged_fn,
    )
    for chromo in range(1, 23):
        bfile = BFILE.format(chromo=chromo)
        out_fn = join(direc, f"clumped_{chromo}")

        cmd = f"{PLINK1} --bfile {bfile} --clump {merged_fn} --clump-p1 {p1} --clump-p2 {p2} --clump-r2 {r2} --clump-kb {kb} --threads {threads} --out {out_fn}"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("errors:", error, flush=True)
        print("output:", output, flush=True)
    results = read_clumps(direc=direc)
    results.to_csv(join(direc, "final_clumps.csv"), index=False)
    n_clumps = len(results)
    return results, n_clumps


def read_clumps(direc):
    full_df = None
    for chromo in range(1, 23):
        clump_fn = join(direc, f"clumped_{chromo}.clumped")
        if os.path.isfile(clump_fn):
            print(f"reading file {clump_fn}")
            df = pd.read_csv(join(direc, f"clumped_{chromo}.clumped"), sep="\s+")
            if full_df is None:
                full_df = df
            else:
                full_df = pd.concat([full_df, df])
    if full_df is None:
        full_df = pd.DataFrame()

    return full_df


def merge_plink(
    out_fn=join("gwas_results", "gwas_merged.txt"), templ=join("gwas_results", TEMPL)
):
    df = get_plink_results(templ=templ)
    df["SNP"] = [line["ID"] for _, line in df.iterrows()]
    df[["SNP", "P"]].to_csv(out_fn, header=True, index=False, sep=" ")


def get_plink_results(templ):
    cols = ["#CHROM", "POS", "ID"]
    for chromo in tqdm(range(1, 23)):
        for pc in range(10):
            sub_df = pd.read_csv(templ.format(chromo=chromo, pc=pc), sep="\t")
            if pc == 0:
                df = sub_df[cols]
            df[f"P_PC{pc}"] = sub_df.P

        if chromo == 1:
            full_df = df
        else:
            full_df = pd.concat([full_df, df])
    full_df["P"] = (10 * full_df.loc[:, [f"P_PC{i}" for i in range(10)]].min(1)).clip(
        1e-320, 1
    )
    return full_df


def run_plink(
    pheno_fn,
    covar_fn,
    out="gwas_results_chr{chromo}",
    threads=20,
):
    for chromo in range(1, 23):
        bfile = BFILE.format(chromo=chromo)
        out_fn = out.format(chromo=chromo)
        print(f"running GWAS on chromo {chromo}", flush=True)
        cmd = f"{PLINK2} --bfile {bfile} --linear hide-covar --covar {covar_fn} --pheno {pheno_fn} --threads {threads} --allow-no-sex --out {out_fn}"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("errors:", error, flush=True)
        print("output:", output, flush=True)


@torch.no_grad()
def export_embeddings(
    loader,
    model,  # ModelCLR
    pheno_fn="tmp.txt",
    cov_fn="cov.txt",
    comp=10,
    dev="cuda:0",
    use_INT=False,
):
    feats = []
    covs = []
    cov_inds = [loader.dataset.cov_columns.index(trait) for trait in COVARIATES]
    iids = []
    for imgs, cov, iid in tqdm(loader):
        batch_embedding = model(imgs.to(dev)).cpu()
        feats.append(batch_embedding)
        covs.append(cov[:, cov_inds])
        iids.append(iid)
    covs = torch.cat(covs).double().numpy()
    covs[:, COVARIATES.index("sex")] += 1
    feats = torch.cat(feats).double().numpy()
    iids = torch.cat(iids).numpy()

    pca = PCA(n_components=comp)
    feats = pca.fit_transform(feats)

    cov = pd.DataFrame(
        {
            "FID": iids,
            "IID": iids,
            **dict((covariate, covs[:, i]) for i, covariate in enumerate(COVARIATES)),
        }
    )
    cov.sex = cov.sex.astype(int)
    cov.to_csv(cov_fn, header=True, index=False, sep="\t")

    df = pd.DataFrame(
        {
            "FID": iids,
            "IID": iids,
            **dict((f"PC{i}", feats[:, i]) for i in range(comp)),
        }
    )
    if use_INT:
        df = inverse_rank_transform(df, cov, method="adjusted")

    df.to_csv(pheno_fn, header=True, index=False, sep="\t")
    return feats, iids


def get_gwas_data(
    seed=42,
    num_workers=8,
    size=256,
    normalize_features=True,
    batch_size=50,
    train_pct=0.7,
    val_pct=0.2,
    cov_fillna="mean",
    return_iid=False,
    eye="left",
    subset=None,
):
    t_iids, v_iids, tt_iids = get_indiv_split(
        train_pct=train_pct, val_pct=val_pct, seed=seed
    )
    loaders = []
    tfms = get_tfms(size=size, augmentation=False)
    for iids, mode in [(t_iids, "train"), (v_iids, "valid"), (tt_iids, "test")]:
        dset = UKBRetina(
            eye=eye,
            iid_selection=iids,
            tfms=tfms,
            normalize_features=normalize_features,
            cov_fillna=cov_fillna,
            return_iid=return_iid,
            subset=subset,
        )
        dset = prune_iids(dset)

        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    return loaders


def prune_iids(dset):
    """make sure each iid only occurs once"""
    unique_iids = set(dset.iids)
    used_iids = set()
    paths = []
    iids = []
    for iid, path in tqdm(zip(dset.iids, dset.paths)):
        if not iid in used_iids:
            paths.append(path)
            iids.append(iid)
            used_iids.add(iid)
    dset.iids = iids
    dset.paths = paths
    return dset


def inverse_rank_transform(df, cov, covars=None, qcovars=None, method="adjusted"):
    pcs = range(df.shape[1] - 2)
    if method == "adjusted":
        cov.index = cov.IID
        cov = cov.loc[df.IID]
        cov = cov.drop(["IID", "FID"], 1)

        df.index = df.IID
        ind = np.intersect1d(cov.index, df.index)
        cov = cov.loc[ind]
        df = df.loc[ind]

        df_adj = df.copy()

        for pc in tqdm(pcs):
            col = f"PC{pc}"
            lr = LinearRegression()
            df_adj[col] = df[col] - lr.fit(cov, df[col]).predict(cov)
        df = df_adj
    for pc in tqdm(pcs):
        col = f"PC{pc}"
        df[col] = INT(df[col])
    return df


def INT(x, method="average", c=3.0 / 8):
    """perform rank-based inverse normal transform"""
    r = stats.rankdata(x, method=method)
    x = (r - c) / (len(x) - 2 * c + 1)
    norm = stats.norm.ppf(x)
    return norm
