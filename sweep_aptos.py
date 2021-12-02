import os
import warnings

import numpy as np
import pytorch_lightning as pl
import toml
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import cohen_kappa_score, accuracy_score
from torch import nn, optim
from torchvision import models

from data.data_aptos import get_aptos_loaders

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CHECKPOINT_PATH = None
CHECKPOINTS_BASE_PATH = toml.load("paths.toml")["CHECKPOINTS_BASE_PATH"]
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_risks_burdens_inner_none/model-epoch_99-valid_loss_5.68.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_snps_gen_none/model_100.pth"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_snps_gen_h1/model-epoch_99-valid_loss_6.62.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_raw_snps_gen_h12/model-epoch_99-valid_loss_5.20.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_risk_scores_gen_none/model-epoch_99-valid_loss_6.28.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_risk_scores_gen_h1/model-epoch_99-valid_loss_6.15.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_risk_scores_gen_h12/model-epoch_99-valid_loss_5.74.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_burden_scores_gen_none/model-epoch_99-valid_loss_5.29.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_burden_scores_gen_h1/model-epoch_99-valid_loss_4.61.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "cm_r50_burden_scores_gen_h12/model-epoch_99-valid_loss_4.93.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "barlow_r50_proj128/epoch_99-step_170399.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "byol_r50_proj128/epoch_99-step_170399.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "simsiam_r50_proj128/epoch_99-step_170399.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "simclr_r50_proj128/epoch_99-step_170399.ckpt"
# CHECKPOINT_PATH = CHECKPOINTS_BASE_PATH + "nnclr_r50_proj128/epoch_99-step_170399.ckpt"

PROJECT_NAME = "aptos-sweep"
defaults = {
    "batch_size": 32,
    "epochs": 10,
    "img_size": 448,
    "accumulate_grad_batches": 1,
    "scheduler": "none",
    "lr": 1e-3,
    "tfms": "default",
}
wandb.init(config=defaults)
config = wandb.config


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
        checkpoint=CHECKPOINT_PATH,
    ):
        super().__init__()
        self.lr = lr
        self.total_steps = total_steps
        self.loss_fct = loss_fct
        self.set_scheduler = set_scheduler
        if checkpoint is None:
            self.model = base_model(pretrained=pretrained)
        else:
            self.model = base_model(pretrained=pretrained)
            state_dict = torch.load(checkpoint, map_location=DEVICE)
            if (
                "simclr" in checkpoint
                or "byol" in checkpoint
                or "barlow" in checkpoint
                or "simsiam" in checkpoint
                or "nnclr" in checkpoint
            ):
                load_from_state_dict_img_only(self.model, state_dict["state_dict"])
            else:
                if "state_dict" in state_dict:
                    load_from_state_dict_gen_img(self.model, state_dict["state_dict"])
                else:
                    load_from_state_dict_gen_img(self.model, state_dict)

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


def main():
    print(config)
    bs = config["batch_size"]
    max_bs = 16

    if bs > max_bs:
        accumulate_grad_batches = int(np.ceil(bs / max_bs))
        bs = bs // accumulate_grad_batches
        print(
            f"set batch_size to {bs} and use accumulate_grad_batches every {accumulate_grad_batches}"
        )
    else:
        accumulate_grad_batches = 1

    tl, vl, ttl = get_aptos_loaders(
        num_workers=10,
        size=config["img_size"],
        batch_size=bs,
    )
    n_classes = 5
    ep = config["epochs"]
    loss_fct = torch.nn.BCEWithLogitsLoss()
    optimizer = "adam"
    optimizer_dict = dict(weight_decay=config["weight_decay"])
    basemodel = models.resnet50
    model = Model(
        n_classes,
        loss_fct=loss_fct,
        base_model=basemodel,
        lr=config["lr"],
        pretrained=False,
        opt_method=optimizer,
        opt_param=optimizer_dict,
        checkpoint=CHECKPOINT_PATH,
    )
    logger = WandbLogger(project=PROJECT_NAME)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=ep,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(model, tl, vl)


if __name__ == "__main__":
    main()
