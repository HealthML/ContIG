import os

import pytorch_lightning as pl
import torch
from PIL import ImageFile
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from data.data_ukb import (
    get_genetics_imaging_data,
    get_pgs_imaging_data,
    get_multimodal_pretraining_data,
    get_imaging_card_data,
)
from models.cross_modal_loss import NTXentLoss
from models.cross_modal_model import ModelCLR

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ImageFile.LOAD_TRUNCATED_IMAGES = True

pl.seed_everything(42)

IMG_SIZE = 448
BATCH_SIZE = 64
ACCUMULATE_GRAD_BATCHES = 1
LR = 1e-3
WEIGHT_DECAY = 1e-6
TEMPERATURE = 0.1
ALPHA_WEIGHT = 0.75
EPOCHS = 101
EVAL_EVERY_N_EPOCHS = 5
RESNET_MODEL_NAME = "resnet50"  # can be "resnet18" or "resnet50"
GENETICS_MODEL_NAME = None  # can be "H1_2048" or "H12_2048" or None
CM_EMBEDDING_SIZE = 128
COV_NOISE_SIZE = 5
# 'raw_snps' or 'risk_scores' or 'burden_scores' or 'covs'
# or it can be a list of combinations of these modalities ["raw_snps", "risk_scores", "burden_scores"]
GENETICS_MODALITY = "raw_snps"
AGGREGATE_MODALITIES = "inner"  # "inner" or "outer"


device = "cuda" if torch.cuda.is_available() else "cpu"

H1 = None
H2 = None
if GENETICS_MODEL_NAME == "H1_2048":
    H1 = 2048
elif GENETICS_MODEL_NAME == "H12_2048":
    H1 = 2048
    H2 = 2048


class CrossModalModel(pl.LightningModule):
    def __init__(self, input_features_dims):
        super().__init__()
        self.multimodal = isinstance(GENETICS_MODALITY, list)
        print(
            "Creating Cross-Modal CLR model, using "
            + str(RESNET_MODEL_NAME)
            + " and "
            + str(GENETICS_MODEL_NAME)
            + " as feature extractors."
        )

        # ModelCLR Initialize
        if not self.multimodal:
            # if training on img + one another genetics modality
            self.model = ModelCLR(
                gen_input_feats=input_features_dims,
                out_dim=CM_EMBEDDING_SIZE,
                hidden1_size=H1,
                hidden2_size=H2,
                genetics_model_name=GENETICS_MODEL_NAME,
            ).to(device)
        else:
            # if img + multiple genetics modalities
            self.models = nn.ModuleList()
            shared_img_encoder = None
            for modality in GENETICS_MODALITY:
                # ["raw_snps", "risk_scores", "burden_scores"]
                feat_dim = None
                if modality == "raw_snps":
                    feat_dim = input_features_dims["gen"]
                elif modality == "risk_scores":
                    feat_dim = input_features_dims["pgs"]
                elif modality == "burden_scores":
                    feat_dim = input_features_dims["burdens"]
                if len(self.models) > 0:
                    shared_img_encoder = self.models[0].imaging_model
                self.models.append(
                    ModelCLR(
                        gen_input_feats=feat_dim,
                        shared_img_encoder=shared_img_encoder,
                        out_dim=CM_EMBEDDING_SIZE,
                        hidden1_size=H1,
                        hidden2_size=H2,
                        genetics_model_name=GENETICS_MODEL_NAME,
                    ).to(device)
                )

        # loss creation
        self.criterion = NTXentLoss(
            device, BATCH_SIZE, temperature=TEMPERATURE, alpha_weight=ALPHA_WEIGHT
        )

    def forward(self, x):
        if not self.multimodal:
            self.model(x)
        else:
            self.models[0](x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if not self.multimodal:
            if GENETICS_MODALITY == "covs":
                xis, cov = batch
                xis = xis.to(device)
                cov_noise = torch.hstack(
                    [cov, torch.rand(len(cov), COV_NOISE_SIZE).to(device)]
                )
                xjs = cov_noise.to(device)
            else:
                xis, cov, xjs = batch
                xis = xis.to(device)
                xjs = xjs.to(device)

            # get the representations and compute the loss
            zis, zjs = self.model(xis, xjs)  # [N,C]
            loss = self.criterion(zis, zjs)
        else:
            img, cov, gen, pgs, burdens, missing = (
                batch["img"],
                batch["cov"],
                batch["gen"],
                batch["pgs"],
                batch["burdens"],
                batch["missing"],
            )
            xis = img.to(device)
            w_per_modality = 1 / (len(GENETICS_MODALITY))
            loss = 0.0
            for i, (modality, cm_model) in enumerate(
                zip(GENETICS_MODALITY, self.models)
            ):
                xjs = None
                if modality == "raw_snps":
                    xjs = gen.to(device)
                elif modality == "risk_scores":
                    xjs = pgs.to(device)
                elif modality == "burden_scores":
                    xjs = burdens.to(device)
                if torch.any(missing):
                    xjs = torch.nan_to_num(xjs)
                # get the representations and compute the loss
                zis, zjs = cm_model(xis, xjs)  # [N,C]
                mask = torch.tensor(missing[:, i] == 0).to(device)
                if torch.any(mask):
                    cm_loss = self.criterion(zis[mask], zjs[mask])
                    loss += w_per_modality * cm_loss

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.multimodal:
            if GENETICS_MODALITY == "covs":
                xis, cov = batch
                xis = xis.to(device)
                cov_noise = torch.hstack(
                    [cov, torch.rand(len(cov), COV_NOISE_SIZE).to(device)]
                )
                xjs = cov_noise.to(device)
            else:
                xis, cov, xjs = batch
                xis = xis.to(device)
                xjs = xjs.to(device)

            # get the representations and compute the loss
            zis, zjs = self.model(xis, xjs)  # [N,C]
            loss = self.criterion(zis, zjs)
        else:
            img, cov, gen, pgs, burdens, missing = (
                batch["img"],
                batch["cov"],
                batch["gen"],
                batch["pgs"],
                batch["burdens"],
                batch["missing"],
            )
            xis = img.to(device)
            w_per_modality = 1 / (len(GENETICS_MODALITY))
            loss = 0.0
            for i, (modality, cm_model) in enumerate(
                zip(GENETICS_MODALITY, self.models)
            ):
                xjs = None
                if modality == "raw_snps":
                    xjs = gen.to(device)
                elif modality == "risk_scores":
                    xjs = pgs.to(device)
                elif modality == "burden_scores":
                    xjs = burdens.to(device)
                if torch.any(missing):
                    xjs = torch.nan_to_num(xjs)
                # get the representations and compute the loss
                zis, zjs = cm_model(xis, xjs)  # [N,C]
                mask = torch.tensor(missing[:, i] == 0).to(device)
                if torch.any(mask):
                    cm_loss = self.criterion(zis[mask], zjs[mask])
                    loss += w_per_modality * cm_loss

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        global training_set_len
        if not self.multimodal:
            optim = torch.optim.Adam(
                self.model.parameters(),
                LR,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=training_set_len, eta_min=0, last_epoch=-1
            )
            return [optim], [scheduler]
        else:
            optims, schs = list(), list()
            for cm_model in self.models:
                optim = torch.optim.Adam(
                    cm_model.parameters(),
                    LR,
                    weight_decay=WEIGHT_DECAY,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=training_set_len, eta_min=0, last_epoch=-1
                )
                optims.append(optim)
                schs.append(scheduler)
            return optims, schs


# Dataloaders
loaders = None
input_features_sizes = None
if GENETICS_MODALITY == "raw_snps":
    loaders, input_features_sizes = get_genetics_imaging_data(
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=8,
        rsids=None,
        sid_slice=slice(0, None, 100),
        train_pct=0.7,
        val_pct=0.1,
    )
elif GENETICS_MODALITY == "burden_scores":
    loaders, input_features_sizes = get_genetics_imaging_data(
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        rsids=None,
        chromos=None,
        sid_slice=None,
        burdens_zeros=1,
        num_workers=8,
        train_pct=0.7,
        val_pct=0.1,
    )
elif GENETICS_MODALITY == "risk_scores":
    loaders, input_features_sizes = get_pgs_imaging_data(
        normalize_pgs=True,
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=8,
        train_pct=0.7,
        val_pct=0.1,
    )
elif GENETICS_MODALITY == "covs":
    loaders, input_features_sizes = get_imaging_card_data(
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=8,
        train_pct=0.7,
        val_pct=0.1,
    )
    input_features_sizes += COV_NOISE_SIZE
else:
    loaders, input_features_sizes = get_multimodal_pretraining_data(
        # inner (=intersection, no missings) or outer (=union, with missings)
        aggregate_modalities=AGGREGATE_MODALITIES,
        modalities=GENETICS_MODALITY,
        # raw genetics
        gen_sid_slice=slice(0, None, 100),
        # pgs
        normalize_pgs=True,
        # burdens
        burdens_zeros=1,  # filter burden scores by numbers of non-zeros (percentage or absolute)
        # general
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=8,
        train_pct=0.7,
        val_pct=0.1,
    )
tl = loaders[0]
vl = loaders[1]
ttl = loaders[2]
print(
    "training samples "
    + str(len(tl))
    + " val samples "
    + str(len(vl))
    + " test samples "
    + str(len(ttl))
)


training_set_len = len(tl)

model = CrossModalModel(input_features_dims=input_features_sizes)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    deterministic=True,
    gpus=1,
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    check_val_every_n_epoch=EVAL_EVERY_N_EPOCHS,
    callbacks=[
        ModelCheckpoint(
            monitor="valid_loss",
            mode="min",
            filename="model-{epoch:02d}-{valid_loss:.2f}",
            save_last=True,
        ),
    ],
)
trainer.fit(model, tl, vl)
print("Finished Training")

print("Testing the model on the test split...")
result = trainer.test(model, dataloaders=ttl)
print(result)
print("Done.")
