import os
import warnings

import numpy as np
import pytorch_lightning as pl
import toml
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import cohen_kappa_score, accuracy_score
from torch import nn
from torch import optim
from torchvision import models

from data.data_aptos import get_aptos_loaders

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set to False if not using wandb
WANDB = True
if WANDB:
    from pytorch_lightning.loggers import WandbLogger

CHECKPOINT_PATH = None
CHECKPOINTS_BASE_PATH = toml.load("paths.toml")["CHECKPOINTS_BASE_PATH"]
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "supervised_baseline/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_risks_burdens_inner_none/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_risks_burdens_inner_h1/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_risks_burdens_outer_none/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_risks_burdens_outer_h1/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_risks_burdens_outer_h12/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_snps_none/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_snps_h1/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_snps_h12/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_risk_scores_gen_none/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_risk_scores_gen_h1/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_risk_scores_gen_h12/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_burden_scores_gen_none/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_burden_scores_gen_h1/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_burden_scores_gen_h12/last.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "barlow_r50_proj128/epoch_99-step_170399.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "byol_r50_proj128/epoch_99-step_170399.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "simsiam_r50_proj128/epoch_99-step_170399.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "simclr_r50_proj128/epoch_99-step_170399.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "nnclr_r50_proj128/epoch_99-step_170399.ckpt"


FROZEN_ENCODER = False
n_classes = 5
multilabel = True
epochs = 50
lr = 1e-3
train_pct = 0.6
val_pct = 0.8 - train_pct
loader_param = {
    "batch_size": 16,
    "size": 448,
    "multilabel": multilabel,
    "train_pct": train_pct,
    "val_pct": val_pct,
}
pretrained_imagenet = False
use_scheduler = False
accumulate_grad_batches = 4
basemodel = models.resnet50
# patience = 10
# optimizer = "sgd"
# optimizer_dict = dict(weight_decay=5e-4, momentum=0.9, nesterov=True)
optimizer = "adam"
optimizer_dict = dict(weight_decay=1e-5)
pl.seed_everything(42, workers=True)


def transform_multilabel_to_continuous(y, threshold):
    assert isinstance(y, np.ndarray), "invalid y"

    y = y > threshold
    y = y.astype(int).sum(axis=1) - 1
    return y


def score_kappa_aptos(y, y_pred, threshold=0.5):
    y = transform_multilabel_to_continuous(y, threshold)
    y_pred = transform_multilabel_to_continuous(y_pred, threshold)
    return cohen_kappa_score(y, y_pred, labels=[0, 1, 2, 3, 4], weights="quadratic")


def acc_aptos(y, y_pred, threshold=0.5):
    y = transform_multilabel_to_continuous(y, threshold)
    y_pred = transform_multilabel_to_continuous(y_pred, threshold)
    return accuracy_score(y, y_pred)


def load_from_state_dict_supervised(model, state_dict):
    """Loads the model weights from the state dictionary."""
    # step 1: filter state dict
    model_keys_prefixes = []
    for okey, oitem in model.state_dict().items():
        model_keys_prefixes.append(okey.split(".")[0])
    new_state_dict = {}
    index = 0
    for key, item in state_dict.items():
        # remove the "model." prefix from the state dict key
        all_key_parts = [model_keys_prefixes[index]]
        all_key_parts.extend(key.split(".")[2:])
        index += 1
        new_key = ".".join(all_key_parts)
        if new_key in model.state_dict() and "fc" not in new_key:
            new_state_dict[new_key] = item

    # step 2: load from checkpoint
    model.load_state_dict(new_state_dict, strict=False)


def load_from_state_dict_gen_img(model, state_dict):
    """Loads the model weights from the state dictionary."""
    # step 1: filter state dict
    model_keys_prefixes = []
    for okey, oitem in model.state_dict().items():
        model_keys_prefixes.append(okey.split(".")[0])
    new_state_dict = {}
    index = 0
    for key, item in state_dict.items():
        if (
            key.startswith("imaging_model")
            or key.startswith("model.imaging_model")
            or key.startswith("models.0.imaging_model")
        ):
            # remove the "model." prefix from the state dict key
            all_key_parts = [model_keys_prefixes[index]]
            if key.startswith("imaging_model"):
                all_key_parts.extend(key.split(".")[2:])
            elif key.startswith("model.imaging_model"):
                all_key_parts.extend(key.split(".")[3:])
            else:
                all_key_parts.extend(key.split(".")[4:])
            index += 1
            new_key = ".".join(all_key_parts)
            if new_key in model.state_dict():
                new_state_dict[new_key] = item

    # step 2: load from checkpoint
    model.load_state_dict(new_state_dict, strict=False)


def load_from_state_dict_img_only(model, state_dict):
    """Loads the model weights from the state dictionary."""
    # step 1: filter state dict
    model_keys_prefixes = []
    for okey, oitem in model.state_dict().items():
        model_keys_prefixes.append(okey.split(".")[0])
    new_state_dict = {}
    index = 0
    for key, item in state_dict.items():
        if (
            (
                key.startswith("resnet_simclr")
                or key.startswith("resnet_simsiam")
                or key.startswith("resnet_barlow_twins")
                or key.startswith("resnet_byol")
                or key.startswith("resnet_nnclr")
            )
            and "projection" not in key
            and "prediction" not in key
            and "momentum" not in key
        ):
            # remove the "model." prefix from the state dict key
            all_key_parts = [model_keys_prefixes[index]]
            all_key_parts.extend(key.split(".")[3:])
            index += 1
            new_key = ".".join(all_key_parts)
            if new_key in model.state_dict():
                new_state_dict[new_key] = item

    # step 2: load from checkpoint
    model.load_state_dict(new_state_dict, strict=False)


class Model(pl.LightningModule):
    def __init__(
        self,
        n_output,
        loss_fct,
        base_model=models.resnet18,
        pretrained=True,
        lr=1e-3,
        total_steps=0,
        set_scheduler="none",
        opt_method="adam",
        opt_param=dict(),
        metrics=[score_kappa_aptos, acc_aptos],
    ):
        super().__init__()
        self.lr = lr
        self.total_steps = total_steps
        self.loss_fct = loss_fct
        self.set_scheduler = set_scheduler
        if CHECKPOINT_PATH is None:
            self.model = base_model(pretrained=pretrained)
        else:
            self.model = base_model(pretrained=pretrained)
            state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            if (
                "simclr" in CHECKPOINT_PATH
                or "byol" in CHECKPOINT_PATH
                or "barlow" in CHECKPOINT_PATH
                or "simsiam" in CHECKPOINT_PATH
                or "nnclr" in CHECKPOINT_PATH
            ):
                load_from_state_dict_img_only(self.model, state_dict["state_dict"])
            else:
                if "state_dict" in state_dict:
                    load_from_state_dict_gen_img(self.model, state_dict["state_dict"])
                else:
                    load_from_state_dict_gen_img(self.model, state_dict)
            if (
                "simclr" in CHECKPOINT_PATH
                or "byol" in CHECKPOINT_PATH
                or "barlow" in CHECKPOINT_PATH
                or "simsiam" in CHECKPOINT_PATH
                or "nnclr" in CHECKPOINT_PATH
            ):
                load_from_state_dict_img_only(self.model, state_dict["state_dict"])
            elif "supervised" in CHECKPOINT_PATH:
                if "state_dict" in state_dict:
                    load_from_state_dict_supervised(
                        self.model, state_dict["state_dict"]
                    )
                else:
                    load_from_state_dict_supervised(self.model, state_dict)
            else:
                if "state_dict" in state_dict:
                    load_from_state_dict_gen_img(self.model, state_dict["state_dict"])
                else:
                    load_from_state_dict_gen_img(self.model, state_dict)

        if FROZEN_ENCODER and CHECKPOINT_PATH is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, n_output)
        self.opt_method = opt_method
        self.opt_param = opt_param
        self.metrics = metrics

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.opt_method == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr, **self.opt_param)
        elif self.opt_method == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.lr, **self.opt_param)
        else:
            raise NotImplementedError(
                f"optimization method {self.opt_method} not set up"
            )
        if self.set_scheduler == "none":
            return optimizer
        elif self.set_scheduler == "steplr":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        elif self.set_scheduler == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.total_steps,
            )
        return [optimizer], [scheduler]

    def training_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fct(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fct(y_hat, y)
        y_np = y.detach().cpu().numpy()
        y_hat_np = y_hat.detach().cpu().numpy()
        if self.metrics is not None:
            for metric in self.metrics:
                self.log(
                    f"valid_{metric.__name__}",
                    metric(y_np, y_hat_np),
                    on_epoch=True,
                    prog_bar=True,
                )
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fct(y_hat, y)
        y_np = y.detach().cpu().numpy()
        y_hat_np = y_hat.detach().cpu().numpy()
        if self.metrics is not None:
            for metric in self.metrics:
                self.log(
                    f"test_{metric.__name__}",
                    metric(y_np, y_hat_np),
                    on_epoch=True,
                    prog_bar=True,
                )
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss


loaders = get_aptos_loaders(**loader_param)
tl, vl, ttl = loaders
print(
    "training samples "
    + str(len(tl))
    + " val samples "
    + str(len(vl))
    + " test samples "
    + str(len(ttl))
)
if multilabel:
    loss_fct = torch.nn.BCEWithLogitsLoss()
else:
    loss_fct = torch.nn.CrossEntropyLoss()

total_steps = epochs * len(tl) if use_scheduler else 0
model = (
    Model(
        n_classes,
        loss_fct=loss_fct,
        base_model=basemodel,
        lr=lr,
        total_steps=total_steps,
        pretrained=pretrained_imagenet,
        set_scheduler="onecycle" if use_scheduler else "none",
        opt_method=optimizer,
        opt_param=optimizer_dict,
    )
    .cuda()
    .train()
)

logger = None
if WANDB:
    logger = WandbLogger(project="aptos_diabetic_retinopathy")
    params = {
        "epochs": epochs,
        "train_pct": train_pct,
        "lr": lr,
        "use_scheduler": use_scheduler,
        "base_model": basemodel.__name__,
        "img_size": tl.dataset[0][0].shape[-1],
        "bs": tl.batch_size,
        "accumulate_grad_batches": accumulate_grad_batches,
    }
    logger.log_hyperparams(params)

trainer = pl.Trainer(
    gpus=1,
    deterministic=True,
    max_epochs=epochs,
    logger=logger if WANDB else True,
    accumulate_grad_batches=accumulate_grad_batches,
    callbacks=[
        # EarlyStopping(monitor="valid_loss", patience=patience),
        ModelCheckpoint(
            monitor="valid_loss",
            filename="model-{epoch:02d}-{valid_loss:.2f}",
            save_top_k=1,
        ),
    ],
)
trainer.validate(model, dataloaders=vl)
trainer.fit(model, tl, vl)

result = trainer.test(dataloaders=ttl, ckpt_path="best")
print(result)
