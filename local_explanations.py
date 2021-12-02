import os
from os.path import join
from pathlib import Path

import matplotlib as mpl
import numpy as np
import seaborn as sns
import toml
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from data.data_ukb import get_pgs_imaging_data, get_genetics_imaging_data
from feature_explanations import FONT_PROPERTIES, get_pgs_labels, run_explanations
from models.model_loading import LoaderModel

sns.set_style("whitegrid")

# TODO: might need to adapt that
FONT_PROPERTIES = Path(mpl.get_data_path(), "Calibri.ttf")

WEIGHTS_BASE = toml.load("paths.toml")["CHECKPOINTS_BASE_PATH"]
PGS_WEIGHTS = join(WEIGHTS_BASE, "cm_r50_risk_scores_gen_h1/last.ckpt")
RAW_WEIGHTS = join(WEIGHTS_BASE, "cm_r50_raw_snps_h1/last.ckpt")

# TODO: fix iids for which to create local explanations
INDIVIDUALS = [
    "2667657",
]


def final_plots():
    method = "ig-noise"
    od = "local_plots_final"
    bs = 1000
    h1 = 2048
    h2 = None
    explain_indivs_pgs(out_dir=od, top=15, bot=15, bs=bs, method=method, h1=h1, h2=h2)
    explain_indivs_raw(out_dir=od, top=15, bot=15, bs=bs, method=method, h1=h1, h2=h2)


# convenience 1-liners


def explain_indivs_raw(
    iids=INDIVIDUALS,
    out_dir="exp_plots",
    load_result=None,
    imgs=None,
    top=6,
    mid=0,
    bot=6,
    method="ig-noise",
    bs=10,
    h1=2048,
    h2=None,
):
    if load_result is None or imgs is None:
        load_result, imgs = load_raw_indiv(iids, bs=bs, h1=h1, h2=h2)
    explain_indivs_base(
        load_result=load_result,
        imgs=imgs,
        out_dir=out_dir,
        fn_templ="indiv_raw_{iid}.svg",
        top=top,
        mid=mid,
        bot=bot,
        method=method,
    )


def explain_indivs_pgs(
    iids=INDIVIDUALS,
    load_result=None,
    imgs=None,
    top=6,
    mid=0,
    bot=6,
    method="ig-noise",
    bs=10,
    out_dir="exp_plots",
    h1=2048,
    h2=None,
):
    if load_result is None or imgs is None:
        load_result, imgs = load_pgs_indiv(
            indivs=iids,
            weights=PGS_WEIGHTS,
            bs=bs,
            h1=h1,
            h2=h2,
        )
    explain_indivs_base(
        load_result=load_result,
        imgs=imgs,
        out_dir=out_dir,
        fn_templ="indiv_pgs_{iid}.svg",
        top=top,
        mid=mid,
        bot=bot,
        method=method,
    )


def explain_indivs_base(
    load_result,
    imgs,
    out_dir="exp_plots",
    fn_templ="{iid}.pdf",
    top=10,
    mid=10,
    bot=10,
    method="ig-noise",
):
    os.makedirs(out_dir, exist_ok=True)
    M, rI, rG, I, G, iid, L = load_result
    exps = run_explanations(
        img=I,
        gen=G,
        ref_img=rI,
        ref_gen=rG,
        model=M,
        methods=[method],
    )
    for i, img in enumerate(imgs):
        fn = join(out_dir, fn_templ.format(iid=iid[i]))
        plot_local_exp(
            img=img,
            exp=exps[method][i],
            labels=L,
            fn=fn,
            top=top,
            bot=bot,
            mid=mid,
        )


# plotting


def plot_local_exp(
    exp,
    labels,
    fn="tmp.pdf",
    top=10,
    bot=10,
    figsize=(5, 10),
):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(10, 1)
    gs.update(wspace=0.01, hspace=0.31)
    ax2 = fig.add_subplot(gs[:, 0])
    asort = exp.argsort()

    cmap1 = sns.cubehelix_palette(
        start=-0.2, rot=0.6, reverse=True, light=0.7, dark=0.25, as_cmap=True
    )
    cmap2 = sns.cubehelix_palette(
        start=0.5, rot=-0.5, reverse=False, light=0.7, dark=0.25, as_cmap=True
    )
    exp_bot = exp[asort[:bot]]
    ax2.barh(
        np.arange(bot),
        exp_bot,
        label="bottom",
        color=[cmap1(x) for x in np.linspace(0, 1, bot)],
    )

    ax2.barh([bot], 0)

    exp_top = exp[asort[-top:]]
    ax2.barh(
        np.arange(bot + 1, bot + 1 + top),
        exp_top,
        label="top",
        color=[cmap2(x) for x in np.linspace(0, 1, top)],
    )

    xticks = np.arange(bot + 1 + top)
    xticklabels = (
        [l[:25] for l in np.array(labels)[asort[:bot]]]
        + ["..."]
        + [l[:25] for l in np.array(labels)[asort[-top:]]]
    )

    plt.xticks(fontsize=19, fontproperties=FONT_PROPERTIES)
    ax2.set_yticks(xticks)
    ax2.set_yticklabels(xticklabels, fontsize=19, fontproperties=FONT_PROPERTIES)
    ax2.set_xlabel("Attribution score", fontsize=25, fontproperties=FONT_PROPERTIES)
    ax2.set_ylabel("Feature", fontsize=25, fontproperties=FONT_PROPERTIES)
    ax2.invert_yaxis()

    plt.savefig(fn, bbox_inches="tight", dpi=300)


# data loading


def load_raw_indiv(
    indivs,
    bs=10,
    weights=RAW_WEIGHTS,
    dev="cuda:0",
    sid_slice=slice(0, None, 100),
    h1=2048,
    h2=None,
):
    (tl, vl, ttl), nf = get_genetics_imaging_data(
        rsids=None,
        sid_slice=sid_slice,
        burdens_zeros=None,
        size=448,
        batch_size=bs,
        return_iid=True,
    )
    labels = tl.dataset.datasets[0].feature_names
    return load_indiv_base(
        indivs=indivs,
        weights=weights,
        tl=tl,
        vl=vl,
        ttl=ttl,
        nf=nf,
        dev=dev,
        h1=h1,
        h2=h2,
        labels=labels,
    )


def load_pgs_indiv(indivs, bs=10, weights=PGS_WEIGHTS, h1=2048, h2=None, dev="cuda:0"):
    (tl, vl, ttl), nf = get_pgs_imaging_data(size=448, batch_size=bs, return_iid=True)
    labels = get_pgs_labels()
    return load_indiv_base(
        indivs=indivs,
        weights=weights,
        tl=tl,
        vl=vl,
        ttl=ttl,
        nf=nf,
        dev=dev,
        labels=labels,
        h1=h1,
        h2=h2,
    )


def load_indiv_base(
    indivs, weights, tl, vl, ttl, nf, h1=2048, h2=None, dev="cuda:0", labels=None
):

    ref_img, _, ref_gen, _ = next(iter(ttl))

    all_iids = [dl.dataset.datasets[0].iids for dl in [tl, vl, ttl]]
    idxs = dict()
    for j, iids in enumerate(all_iids):
        for iid in indivs:
            if iid in iids:
                idxs[iid] = (j, list(iids).index(iid))
    imgs = []
    timgs = []
    gens = []
    for iid in indivs:
        dset = [tl, vl, ttl][idxs[iid][0]].dataset.datasets[0]
        idx = idxs[iid][1]
        t, _, gen, _ = dset[idx]
        img = Image.open(dset.paths[idx])
        timgs.append(t)
        imgs.append(img)
        gens.append(gen)
    timgs = torch.stack(timgs)
    gens = torch.stack(gens)

    LM = LoaderModel(nf, h1=h1, h2=h2)
    LM.init(weights)
    ret = (LM.model.eval().to(dev), ref_img, ref_gen, timgs, gens, indivs, labels)
    return ret, imgs
