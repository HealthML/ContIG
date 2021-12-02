import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("whitegrid")

from torch import nn

from data.data_ukb import *

from models.cross_modal_loss import NTXentLoss
from models.model_loading import LoaderModel, LoaderModelCrossModal

from captum import attr


# TODO: might need to adapt that
FONT_PROPERTIES = Path(mpl.get_data_path(), "Calibri.ttf")

WEIGHTS_BASE = toml.load("paths.toml")["CHECKPOINTS_BASE_PATH"]


COV_WEIGHTS = join(WEIGHTS_BASE, "cm_r50_cov_gen_none/last.ckpt")
RAW_WEIGHTS = join(WEIGHTS_BASE, "cm_r50_raw_snps_h1/last.ckpt")
PGS_WEIGHTS = join(WEIGHTS_BASE, "cm_r50_risk_scores_gen_h1/last.ckpt")
BURDEN_WEIGHTS = join(WEIGHTS_BASE, "cm_r50_burden_scores_gen_h1/last.ckpt")
MULTI_WEIGHTS = join(
    WEIGHTS_BASE, "cm_r50_raw_risks_burdens_outer_h1/checkpoints/last.ckpt"
)

INPUT_FEATURE_DIMS = {"pgs": 481, "gen": 7854, "burdens": 18574}

BS = 1000

METHODS = [
    # "ig",
    "ig-noise",
    # "dl",
]


# convenience 1-liners


def final_plots():
    methods = ["ig-noise"]
    od = "exp_plots_final"
    explain_covariates(out_dir=od, methods=methods, h1=None, h2=None)
    explain_pgs(out_dir=od, methods=methods, h1=2048, h2=None)
    explain_raw_snps(out_dir=od, methods=methods, h1=2048, h2=None)
    explain_burden(out_dir=od, methods=methods, h1=2048, h2=None)


def explain_burden(
    load_burden_ret=None,
    out_dir="exp_plots",
    top=30,
    bot=0,
    methods=METHODS,
    h1=2048,
    h2=None,
):
    print("loading data ...")
    if load_burden_ret is None:
        load_burden_ret = load_burden_base(weights=BURDEN_WEIGHTS, bs=BS, h1=h1, h2=h2)
    return explain_gen_base(
        load_burden_ret,
        name="burden",
        out_dir=out_dir,
        top=top,
        bot=bot,
        methods=methods,
    )


def explain_raw_snps(
    load_raw_ret=None,
    out_dir="exp_plots",
    top=30,
    bot=0,
    methods=METHODS,
    h1=2048,
    h2=None,
):
    print("loading data ...")
    if load_raw_ret is None:
        load_raw_ret = load_raw_snps_base(weights=RAW_WEIGHTS, bs=BS, h1=h1, h2=h2)
    return explain_gen_base(
        load_raw_ret,
        name="raw-snps",
        out_dir=out_dir,
        top=top,
        bot=bot,
        methods=methods,
    )


def explain_pgs(
    load_pgs_ret=None,
    out_dir="exp_plots",
    top=30,
    bot=0,
    methods=METHODS,
    h1=2048,
    h2=None,
):
    """use `load_pgs_ret` as return values of `load_pgs_base` for faster loading"""
    print("loading data ...")
    if load_pgs_ret is None:
        load_pgs_ret = load_pgs_base(weights=PGS_WEIGHTS, bs=BS, h1=h1, h2=h2)
    return explain_gen_base(
        load_pgs_ret, name="pgs", out_dir=out_dir, top=top, bot=bot, methods=methods
    )


def explain_covariates(
    load_cov_ret=None, out_dir="exp_plots", methods=METHODS, h1=None, h2=None
):
    """use `load_cov_ret` as return values of `load_cov_base` for faster loading"""
    print("loading data ...")
    if load_cov_ret is None:
        load_cov_ret = load_cov_base(
            weights=COV_WEIGHTS, bs=BS, noise_dim=5, h1=h1, h2=h2
        )
    return explain_gen_base(
        load_cov_ret,
        name="cov",
        out_dir=out_dir,
        top=0,
        bot=0,
        methods=methods,
        h=False,
    )


def explain_multimodal(
    load_mm_ret,
    out_dir="exp_plots",
    top=10,
    bot=10,
    h1=2048,
    h2=None,
    methods=["ig-noise"],
):
    if load_mm_ret is None:
        load_mm_ret = load_multimodal_base(weights=MULTI_WEIGHTS, bs=BS, h1=h1, h2=h2)
    M, rI, rG, rP, rB, I, G, P, B, iid, L = load_mm_ret
    print("start explanations...")
    exps = run_multimodal_explanations(
        img=I,
        gen=G,
        pgs=P,
        burdens=B,
        ref_img=rI,
        ref_gen=rG,
        ref_pgs=rP,
        ref_burdens=rB,
        model=M,
        methods=methods,
    )
    print("plotting...")
    os.makedirs(out_dir, exist_ok=True)

    name = f"multimodal-{h1}-{h2}"
    pickle.dump(
        [exps, L, iid],
        open(join(out_dir, f"{name}_exp_{len(list(exps.values())[0])}.pkl"), "wb"),
    )

    if top > 0 or bot > 0:
        top_inds = get_top_explanations(exps, top=top, bot=bot)
        exps = dict((k, v[:, top_inds]) for k, v in exps.items())
        L = [f"{L[i]}-{i}" for i in top_inds]

    plot_global_hist_single(exps[methods[0]], L, fn=join(out_dir, f"{name}_bars.pdf"))
    plot_global_hist(exps, L, fn=join(out_dir, f"{name}_hist.png"))
    plot_global_box(exps, L, fn=join(out_dir, f"{name}_box.png"))
    return exps, L


def explain_multimodal_aggregated(out_dir="exp_plots", bs=1000):
    print("loading")
    exps, L, iid = pickle.load(
        open(join(out_dir, f"multimodal-2048-None_exp_{bs}.pkl"), "rb")
    )
    print("aggregating")
    E = np.abs(exps[list(exps.keys())[0]]).mean(0)

    print("computing")
    f = INPUT_FEATURE_DIMS["gen"]
    ff = f + INPUT_FEATURE_DIMS["pgs"]
    gen = E[:f]
    pgs = E[f:ff]
    burdens = E[ff:]

    gen_attribution = gen.sum()
    pgs_attribution = pgs.sum()
    burdens_attribution = burdens.sum()

    gen_mean = gen.mean()
    pgs_mean = pgs.mean()
    burdens_mean = burdens.mean()
    gen_std = gen.std()
    pgs_std = pgs.std()
    burdens_std = burdens.std()

    plot_global_hist_single(
        exp=np.array([[gen_attribution, pgs_attribution, burdens_attribution]]),
        exp_labels=["Raw-SNPs (sum)", "PGS (sum)", "Burdens (sum)"],
        figsize=(5, 5),
        fn=join(out_dir, "aggregated_multimodal_exp_sum.pdf"),
        log_scale=False,
        sort=False,
        xaxis_title="Absolute attribution (sum)",
    )

    plot_global_hist_single(
        exp=np.array([[gen_mean, pgs_mean, burdens_mean]]),
        exp_labels=["Raw-SNPs (mean)", "PGS (mean)", "Burdens (mean)"],
        figsize=(5, 5),
        fn=join(out_dir, "aggregated_multimodal_exp_mean.pdf"),
        log_scale=False,
        sort=False,
        xaxis_title="Absolute attribution (mean)",
    )

    return {
        "gen": (gen_attribution, gen_mean, gen_std),
        "pgs": (pgs_attribution, pgs_mean, pgs_std),
        "burdens": (burdens_attribution, burdens_mean, burdens_std),
    }


def explain_gen_base(
    load_result,
    name="pgs",
    methods=METHODS,
    out_dir="exp_plots",
    top=30,
    bot=0,
    h=True,
):
    M, rI, rG, I, G, iid, L = load_result
    print("start explanations...")
    exps = run_explanations(
        img=I,
        gen=G,
        ref_img=rI,
        ref_gen=rG,
        model=M,
        methods=methods,
    )
    print("plotting...")
    os.makedirs(out_dir, exist_ok=True)

    pickle.dump(
        [exps, L, iid],
        open(join(out_dir, f"{name}_exp_{len(list(exps.values())[0])}.pkl"), "wb"),
    )

    if top > 0 or bot > 0:
        top_inds = get_top_explanations(exps, top=top, bot=bot)
        exps = dict((k, v[:, top_inds]) for k, v in exps.items())
        L = [f"{L[i][:50]}-{i}" for i in top_inds]
        # L = [f"{L[i]}-{i}" for i in top_inds]

    plot_global_hist_single(
        exps[methods[0]], L, fn=join(out_dir, f"{name}_bars.pdf"), h=h
    )
    plot_global_hist(exps, L, fn=join(out_dir, f"{name}_hist.png"))
    plot_global_box(exps, L, fn=join(out_dir, f"{name}_box.png"))
    return exps, L


def load_results_and_plot(path, out_dir, name, top=30, bot=0, methods=["ig-noise"]):
    exps, L, iid = pickle.load(open(path, "rb"))
    if top > 0 or bot > 0:
        top_inds = get_top_explanations(exps, top=top, bot=bot)
        exps = dict((k, v[:, top_inds]) for k, v in exps.items())
        Li = [f"{L[i]}-{i}" for i in top_inds]
        L = [L[i] for i in top_inds]

    plot_global_hist_single(exps[methods[0]], L, fn=join(out_dir, f"{name}_bars.pdf"))
    plot_global_hist(exps, L, fn=join(out_dir, f"{name}_hist.png"))
    plot_global_box(exps, Li, fn=join(out_dir, f"{name}_box.png"))


## attribution


def run_multimodal_explanations(
    img,
    gen,
    pgs,
    burdens,
    ref_img,
    ref_gen,
    ref_pgs,
    ref_burdens,
    model,
    temperature=1.0,
    alpha=0.5,
    methods=["ig-noise"],
):
    explainer = MultiGeneticsExplainer(
        model=model,
        ref_img=ref_img,
        ref_gen=ref_gen,
        ref_pgs=ref_pgs,
        ref_burdens=ref_burdens,
        temperature=temperature,
        alpha=alpha,
    )
    G = torch.cat([gen, pgs, burdens], dim=-1)
    exps = dict()

    if "ig" in methods:
        print("computing integrated gradients...")
        ig_attr_test = explain_multiple(
            gen=G,
            img=img,
            explainer=explainer,
            attr_method=attr.IntegratedGradients,
            attr_kwargs={"n_steps": 50, "internal_batch_size": 10},
        )
        exps["ig"] = ig_attr_test

    if "ig-noise" in methods:
        print("computing integrated gradients with noise tunnel...")
        ig_nt_attr_test = explain_multiple(
            gen=G,
            img=img,
            explainer=explainer,
            attr_method=lambda e: attr.NoiseTunnel(attr.IntegratedGradients(e)),
            attr_kwargs={"n_steps": 50, "internal_batch_size": 10},
        )
        exps["ig-noise"] = ig_nt_attr_test

    if "dl" in methods:
        print("computing deep lift...")
        dl_attr_test = explain_multiple(
            gen=G,
            img=img,
            explainer=explainer,
            attr_method=attr.DeepLift,
        )
        exps["dl"] = dl_attr_test

    return exps


def run_explanations(
    img, gen, ref_img, ref_gen, model, methods=["ig"], temperature=1.0, alpha=0.5
):
    explainer = GeneticsExplainer(
        model=model,
        ref_img=ref_img,
        ref_gen=ref_gen,
        temperature=1.0,
        alpha=0.5,
    )
    exps = dict()

    if "ig" in methods:
        print("computing integrated gradients...")
        ig_attr_test = explain_multiple(
            gen=gen,
            img=img,
            explainer=explainer,
            attr_method=attr.IntegratedGradients,
            attr_kwargs={"n_steps": 50, "internal_batch_size": 10},
        )
        exps["ig"] = ig_attr_test

    if "ig-noise" in methods:
        print("computing integrated gradients with noise tunnel...")
        ig_nt_attr_test = explain_multiple(
            gen=gen,
            img=img,
            explainer=explainer,
            attr_method=lambda e: attr.NoiseTunnel(attr.IntegratedGradients(e)),
            attr_kwargs={"n_steps": 50, "internal_batch_size": 10},
        )
        exps["ig-noise"] = ig_nt_attr_test

    if "dl" in methods:
        print("computing deep lift...")
        dl_attr_test = explain_multiple(
            gen=gen,
            img=img,
            explainer=explainer,
            attr_method=attr.DeepLift,
        )
        exps["dl"] = dl_attr_test

    return exps


## plotting


def plot_top(E, L, top=20, bot=0):
    inds = get_top_explanations(E, top)
    E = [e[:, inds] for e in E]
    L = [L[i] for i in inds]
    plot_global_hist(E, L, abs_val=True)
    plot_global_box(E, [f"{i}-{l}" for i, l in enumerate(L)])


def plot_global_hist_single(
    exp,
    exp_labels=None,
    figsize=(5, 10),
    fn="explanations_bar.png",
    log_scale=False,
    h=True,
    sort=True,
    xaxis_title="Mean absolute attribution",
):
    fig = plt.figure(figsize=figsize if h else (figsize[1] * 2, figsize[0]))
    df = pd.DataFrame(exp, columns=exp_labels)
    print(df)
    if sort:
        x = np.argsort(df.abs().mean())[::-1]
        df = df.iloc[:, x.values]

    sns.set_palette(
        sns.cubehelix_palette(
            start=0.5,
            rot=-0.5,
            n_colors=len(exp_labels),
            reverse=True,
            light=0.7,
            dark=0.25,
        ),
        n_colors=len(exp_labels),
    )
    if h:
        ax = sns.barplot(
            y=np.arange(df.shape[1]), x=df.abs().mean(), ci=None, orient="h"
        )

        if log_scale:
            ax.set_xscale("log")

        plt.yticks(fontsize=19, fontproperties=FONT_PROPERTIES)
        ax.set_yticklabels(
            [l[:25] for l in df.columns], fontsize=19, fontproperties=FONT_PROPERTIES
        )
        plt.xticks(fontsize=19, fontproperties=FONT_PROPERTIES)
        ax.set_ylabel("Feature", fontsize=25, fontproperties=FONT_PROPERTIES)
        ax.set_xlabel(xaxis_title, fontsize=25, fontproperties=FONT_PROPERTIES)
    else:
        ax = sns.barplot(
            x=np.arange(df.shape[1]),
            y=df.abs().mean(),
            ci=None,
            orient="v",
        )

        if log_scale:
            ax.set_yscale("log")

        plt.xticks(fontsize=19, fontproperties=FONT_PROPERTIES, rotation="vertical")
        ax.set_xticklabels(
            [l[:25] for l in df.columns], fontsize=19, fontproperties=FONT_PROPERTIES
        )
        plt.yticks(fontsize=19, fontproperties=FONT_PROPERTIES)
        ax.set_xlabel("Feature", fontsize=25, fontproperties=FONT_PROPERTIES)
        ax.set_ylabel(
            "Mean absolute attribution", fontsize=25, fontproperties=FONT_PROPERTIES
        )
    plt.savefig(fn, bbox_inches="tight")


def plot_global_hist(
    exps,
    exp_labels=None,
    figsize=(20, 10),
    abs_val=True,
    fn="explanations_bar.png",
):
    legends = []
    plt.figure(figsize=figsize)
    ax = plt.subplot()
    width = 1 / (len(exps) + 1)
    x_axis_data = np.arange(list(exps.values())[0].shape[1])
    for i, exp_key in enumerate(exps):
        legends.append(exp_key)
        exp = exps[exp_key]
        exp = np.abs(exp).mean(0) if abs_val else exp.mean(0)

        ax.bar(x_axis_data + (i + 0.5) * width, exp, width, align="center")
    if exp_labels is not None:
        ax.set_xticks(x_axis_data + 0.5)
        ax.set_xticklabels([l[:50] for l in exp_labels], rotation="vertical")
    plt.legend(legends, loc="best")

    ax.autoscale_view()
    plt.tight_layout()

    plt.savefig(fn)
    plt.show()


def plot_global_box(
    exps,
    exp_labels=None,
    figsize=(20, 30),
    fn="explanations_box.png",
):
    legends = []
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    for i, exp_key in enumerate(exps):
        legends.append(exp_key)
        exp = exps[exp_key]
        ax = axes[i]
        ax.set_title(legends[i])
        sns.boxenplot(
            data=pd.DataFrame(exp, columns=exp_labels),
            ax=ax,
        )
        ax.autoscale_view()
        if exp_labels is not None:
            ax.set_xticklabels([l[:30] for l in exp_labels], rotation="vertical")
    plt.tight_layout()

    plt.savefig(fn)
    plt.show()


## explanations


def explain_multiple(
    gen, img, explainer, attr_method, attr_kwargs=dict(), dev="cuda:0"
):
    attr_tests = []
    for i in tqdm(range(len(gen))):
        explainer.set_img_embedding(img[i].to(dev))
        attr_test = attr_method(explainer).attribute(
            gen[i : (i + 1)].to(dev), **attr_kwargs
        )
        attr_tests.append(attr_test.detach().cpu().double().numpy())
    return np.concatenate(attr_tests)


class MultiGeneticsExplainer(nn.Module):
    """explanation utility for multi-modal genetic model"""

    def __init__(
        self,
        model,
        ref_img,
        ref_gen,
        ref_pgs,
        ref_burdens,
        temperature=1.0,
        alpha=0.5,
        dev="cuda:0",
    ):
        super().__init__()
        self.model = model.eval()
        with torch.no_grad():
            self.ref_img_embedding = evaluate_in_batches(
                model=self.model.models[0].image_encoder,
                tensor=ref_img,
                bs=2,
                grad=False,
                dev=dev,
            ).to(dev)

            self.ref_gen_embedding = evaluate_in_batches(
                model=self.model.models[0].genetics_encoder,
                tensor=ref_gen,
                bs=10,
                grad=False,
                dev=dev,
            ).to(dev)
            self.ref_pgs_embedding = evaluate_in_batches(
                model=self.model.models[1].genetics_encoder,
                tensor=ref_pgs,
                bs=10,
                grad=False,
                dev=dev,
            ).to(dev)
            self.ref_burdens_embedding = evaluate_in_batches(
                model=self.model.models[2].genetics_encoder,
                tensor=ref_burdens,
                bs=10,
                grad=False,
                dev=dev,
            ).to(dev)

        self.feature_dims = [ref_gen.shape[1], ref_pgs.shape[1], ref_burdens.shape[1]]
        loss_func_single = NTXentLoss(
            device=dev,
            batch_size=None,
            temperature=temperature,
            alpha_weight=alpha,
        )
        w = 1 / 3

        def lf(imgE, genE, pgsE, burdenE):
            loss = 0.0
            loss += w * loss_func_single(imgE, genE)
            loss += w * loss_func_single(imgE, pgsE)
            loss += w * loss_func_single(imgE, burdenE)
            return loss

        self.loss_func = lf

    def set_img_embedding(self, img):
        with torch.no_grad():
            self.img_embedding = (
                self.model.models[0].image_encoder(img.unsqueeze(0)).unsqueeze(0)
            )

    def forward(self, gen):
        """only run with gen belonging to the same image; pre-set with set_img_embedding!"""
        loss = torch.zeros(len(gen))
        img_embedding_mod = torch.cat(
            [
                self.ref_img_embedding,
                self.img_embedding,
            ]
        )
        for i in range(len(gen)):
            f = self.feature_dims[0]
            ff = f + self.feature_dims[1]
            gen_embedding_mod = torch.cat(
                [
                    self.ref_gen_embedding,
                    self.model.models[0].genetics_encoder(gen[i, :f].unsqueeze(0)),
                ]
            )
            pgs_embedding_mod = torch.cat(
                [
                    self.ref_pgs_embedding,
                    self.model.models[1].genetics_encoder(gen[i, f:ff].unsqueeze(0)),
                ]
            )
            burdens_embedding_mod = torch.cat(
                [
                    self.ref_burdens_embedding,
                    self.model.models[2].genetics_encoder(gen[i, ff:].unsqueeze(0)),
                ]
            )
            loss[i] = self.loss_func(
                img_embedding_mod,
                gen_embedding_mod,
                pgs_embedding_mod,
                burdens_embedding_mod,
            )
        return loss


class GeneticsExplainer(nn.Module):
    """explanation utility for model with single genetic modality"""

    def __init__(
        self, model, ref_img, ref_gen, temperature=1.0, alpha=0.5, dev="cuda:0"
    ):
        super().__init__()
        self.model = model.eval()
        with torch.no_grad():
            self.ref_img_embedding = evaluate_in_batches(
                self.model.image_encoder,
                ref_img,
                2,
                grad=False,
                dev=dev,
            ).to(dev)
            self.ref_gen_embedding = evaluate_in_batches(
                self.model.genetics_encoder,
                ref_gen,
                10,
                grad=False,
                dev=dev,
            ).to(dev)
        self.loss_func = NTXentLoss(
            device=dev,
            batch_size=None,
            temperature=temperature,
            alpha_weight=alpha,
        )

    def set_img_embedding(self, img):
        with torch.no_grad():
            self.img_embedding = self.model.image_encoder(img.unsqueeze(0)).unsqueeze(0)

    def forward(self, gen):
        """only run with gen belonging to the same image; pre-set with set_img_embedding!"""
        loss = torch.zeros(len(gen))
        img_embedding_mod = torch.cat(
            [
                self.ref_img_embedding,
                self.img_embedding,
            ]
        )
        for i in range(len(gen)):
            gen_embedding_mod = torch.cat(
                [
                    self.ref_gen_embedding,
                    self.model.genetics_encoder(gen[i].unsqueeze(0)),
                ]
            )
            loss[i] = self.loss_func(img_embedding_mod, gen_embedding_mod)
        return loss


## data handling


def load_cov_base(
    weights=COV_WEIGHTS, bs=4, noise_dim=5, dev="cuda:0", h1=None, h2=None
):
    (_, vl, ttl), nf = get_imaging_card_data(size=448, batch_size=bs, return_iid=True)
    nf += noise_dim
    ref_img, ref_cov, _ = next(iter(ttl))
    img, cov, iid = next(iter(vl))
    cov = torch.hstack([cov, torch.rand(bs, noise_dim)])
    ref_cov = torch.hstack([ref_cov, torch.rand(bs, noise_dim)])
    LM = LoaderModel(nf, h1=h1, h2=h2)
    LM.init(weights)
    labels = get_cov_labels(vl)
    return LM.model.eval().to(dev), ref_img, ref_cov, img, cov, iid, labels


def load_pgs_base(weights=PGS_WEIGHTS, bs=4, dev="cuda:0", h1=None, h2=None):
    (_, vl, ttl), nf = get_pgs_imaging_data(size=448, batch_size=bs, return_iid=True)
    labels = get_pgs_labels()
    return (*load_gen_base(weights, vl, ttl, nf, h1=h1, h2=h2, dev=dev), labels)


def load_burden_base(
    weights=BURDEN_WEIGHTS, bs=4, dev="cuda:0", zeros=1, h1=None, h2=None
):
    (_, vl, ttl), nf = get_genetics_imaging_data(
        rsids=None,
        sid_slice=None,
        burdens_zeros=zeros,
        size=448,
        batch_size=bs,
        return_iid=True,
    )
    labels = tl.dataset.datasets[0].feature_names
    return (*load_gen_base(weights, vl, ttl, nf, h1=h1, h2=h2, dev=dev), labels)


def load_raw_snps_base(
    weights=RAW_WEIGHTS,
    sid_slice=slice(0, None, 100),
    bs=4,
    dev="cuda:0",
    h1=None,
    h2=None,
):
    (_, vl, ttl), nf = get_genetics_imaging_data(
        rsids=None,
        sid_slice=sid_slice,
        burdens_zeros=None,
        size=448,
        batch_size=bs,
        return_iid=True,
    )
    labels = vl.dataset.datasets[0].feature_names
    return (*load_gen_base(weights, vl, ttl, nf, h1=h1, h2=h2, dev=dev), labels)


def load_gen_base(weights, vl, ttl, nf, h1=None, h2=None, dev="cuda:0"):
    img, _, gen, iid = next(iter(ttl))
    ref_img, _, ref_gen, _ = next(iter(vl))
    LM = LoaderModel(nf, h1=h1, h2=h2)
    LM.init(weights)
    return LM.model.eval().to(dev), ref_img, ref_gen, img, gen, iid


def load_multimodal_base(weights=MULTI_WEIGHTS, bs=4, dev="cuda:0", h1=2048, h2=None):
    (_, vl, ttl), nf = get_multimodal_pretraining_data(
        aggregate_modalities="inner",
        size=448,
        gen_sid_slice=slice(0, None, 100),
        batch_size=bs,
        return_iid=True,
        burdens_zeros=1,
    )
    gen_labels = vl.dataset.datasets[0].gen_feature_names
    pgs_labels = get_pgs_labels()
    burden_labels = vl.dataset.datasets[0].burdens_feature_names
    labels = gen_labels + pgs_labels + burden_labels

    batch = next(iter(vl))
    ref_img, ref_gen, ref_pgs, ref_burdens = (
        batch["img"],
        batch["gen"],
        batch["pgs"],
        batch["burdens"],
    )
    batch = next(iter(ttl))
    img, gen, pgs, burdens, iids = (
        batch["img"],
        batch["gen"],
        batch["pgs"],
        batch["burdens"],
        batch["iid"],
    )
    LM = LoaderModelCrossModal(input_feature_dims=nf, h1=h1, h2=h2)
    LM.init(weights)
    return (
        LM.eval().to(dev),
        ref_img,
        ref_gen,
        ref_pgs,
        ref_burdens,
        img,
        gen,
        pgs,
        burdens,
        iids,
        labels,
    )


def get_cov_labels(loader, noise_dims=5):
    return loader.dataset.datasets[0].cov_columns + [
        f"noise{i}" for i in range(noise_dims)
    ]


def get_pgs_labels(index=None):
    df = pd.read_csv("pgs_traits.csv", index_col=2)
    if index is not None:
        df = df.iloc[index]
    return list(df.trait.values)


## utils


def evaluate_in_batches(model, tensor, bs=2, grad=False, dev="cuda:0"):
    num_batches = int(np.ceil(len(tensor) / bs))
    all_out = []
    for b in range(num_batches):
        batch = tensor[b * bs : (b + 1) * bs]
        if not grad:
            with torch.no_grad():
                out = model(batch.to(dev)).cpu()
        else:
            out = model(batch.to(dev))
        all_out.append(out)
    return torch.cat(all_out)


def get_top_explanations(exps, top=10, bot=10, abs_val=True):
    exps = np.array(
        [np.abs(exp).mean(0) if abs_val else exp.mean(0) for exp in exps.values()]
    )
    agg_exps = exps.mean(0)
    asort = np.argsort(agg_exps)
    inds = np.concatenate([asort[:bot], asort[-top:]])
    return inds
