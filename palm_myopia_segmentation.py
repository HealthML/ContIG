import os
import warnings

import numpy as np
import pytorch_lightning as pl
import toml
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch import optim
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCNHead, FCN

from data.data_palm import get_palm_loaders
from models.resnet_unet import UNetWithResnet50Encoder

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

train_pct = 0.6
val_pct = 0.8 - train_pct
loader_param = {
    "batch_size": 4,
    "size": 448,
    "joint_mask": True,
    "train_pct": train_pct,
    "val_pct": val_pct,
}
accumulate_grad_batches = 16
n_classes = 2
epochs = 50
warmup_epochs = 10  # if set to 0, fine-tune in all epochs
lr = 1e-3
dice_weight = 0.8
bce_weight = 0.2
seg_model_name = "unet"  # "fcn" or "deeplabv3" or "unet"
basemodel = models.resnet50
pretrained_imagenet = False
set_scheduler = "none"  # "none" or "steplr" or "onecycle" or "reduceplat"
# optimizer = "sgd"
# optimizer_dict = dict(weight_decay=5e-4, momentum=0.9, nesterov=True)
optimizer = "adam"
optimizer_dict = dict(weight_decay=1e-5)

pl.seed_everything(42, workers=True)


def dice(y, y_pred):
    intersection = np.sum(y_pred * y) * 2.0
    return intersection / (np.sum(y_pred) + np.sum(y))


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
        base_model=models.resnet50,
        seg_model_name="fcn",  # can be "fcn" or "deeplabv3" or "unet"
        pretrained=True,
        lr=1e-3,
        total_steps=0,
        set_scheduler="none",
        opt_method="adam",
        opt_param=dict(),
    ):
        super().__init__()
        self.lr = lr
        self.total_steps = total_steps
        self.loss_fct = loss_fct
        self.set_scheduler = set_scheduler
        if CHECKPOINT_PATH is None:
            backbone = base_model(pretrained=pretrained)
        else:
            backbone = base_model(pretrained=pretrained)
            state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            if (
                "simclr" in CHECKPOINT_PATH
                or "byol" in CHECKPOINT_PATH
                or "barlow" in CHECKPOINT_PATH
                or "simsiam" in CHECKPOINT_PATH
                or "nnclr" in CHECKPOINT_PATH
            ):
                load_from_state_dict_img_only(backbone, state_dict["state_dict"])
            elif "supervised" in CHECKPOINT_PATH:
                if "state_dict" in state_dict:
                    load_from_state_dict_supervised(backbone, state_dict["state_dict"])
                else:
                    load_from_state_dict_supervised(backbone, state_dict)
            else:
                if "state_dict" in state_dict:
                    load_from_state_dict_gen_img(backbone, state_dict["state_dict"])
                else:
                    load_from_state_dict_gen_img(backbone, state_dict)

        if warmup_epochs > 0 and CHECKPOINT_PATH is not None:
            for param in backbone.parameters():
                param.requires_grad = False

        if seg_model_name == "fcn" or seg_model_name == "deeplabv3":
            out_layer = "layer4"
            out_inplanes = 2048
            return_layers = {out_layer: "out"}
            backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
            model_map = {
                "deeplabv3": (DeepLabHead, DeepLabV3),
                "fcn": (FCNHead, FCN),
            }
            classifier = model_map[seg_model_name][0](out_inplanes, n_output)
            base_model = model_map[seg_model_name][1]
            self.model = base_model(backbone, classifier, aux_classifier=None)
        else:
            self.model = UNetWithResnet50Encoder(backbone, n_classes=n_output)
        self.opt_method = opt_method
        self.opt_param = opt_param
        self.labels = []
        self.preds = []

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
        elif self.set_scheduler == "reduceplat":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return {
                "optimizer": optimizer,
                "scheduler": scheduler,
                "monitor": "valid_loss",
            }
        return [optimizer], [scheduler]

    def on_train_epoch_start(self) -> None:
        if warmup_epochs > 0 and self.current_epoch == warmup_epochs:
            if CHECKPOINT_PATH is not None:
                for param in self.parameters():
                    param.requires_grad = True
            self.trainer.optimizers[0] = optim.Adam(
                self.parameters(), lr=self.lr / 10, **self.opt_param
            )

    def training_step(self, batch, idx):
        x, y = batch
        if seg_model_name == "fcn" or seg_model_name == "deeplabv3":
            y_hat = self(x)["out"]
        else:
            y_hat = self(x)
        loss = self.loss_fct(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        if seg_model_name == "fcn" or seg_model_name == "deeplabv3":
            y_hat = self(x)["out"]
        else:
            y_hat = self(x)
        loss = self.loss_fct(y_hat, y)
        y_np = y.detach().cpu().numpy()
        y_hat_np = F.sigmoid(y_hat).detach().cpu().numpy()
        self.store_predictions_labels(y_np, y_hat_np)
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True)
        if idx == 0:
            self.display_batch_imgs(x, y_hat_np, y_np, title="val images")
        return loss

    def test_step(self, batch, idx):
        x, y = batch
        if seg_model_name == "fcn" or seg_model_name == "deeplabv3":
            y_hat = self(x)["out"]
        else:
            y_hat = self(x)
        loss = self.loss_fct(y_hat, y)
        y_np = y.detach().cpu().numpy()
        y_hat_np = F.sigmoid(y_hat).detach().cpu().numpy()
        self.store_predictions_labels(y_np, y_hat_np)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.display_batch_imgs(x, y_hat_np, y_np, title="test images")
        return loss

    def on_validation_epoch_end(self) -> None:
        y = np.concatenate(self.labels).ravel()
        y_hat = np.concatenate(self.preds).ravel()
        self.log(
            "valid_dice",
            dice(y, y_hat),
        )
        self.labels = []
        self.preds = []

    def on_test_epoch_end(self) -> None:
        y = np.concatenate(self.labels).ravel()
        y_hat = np.concatenate(self.preds).ravel()
        self.log(
            "test_dice",
            dice(y, y_hat),
        )
        self.labels = []
        self.preds = []

    def store_predictions_labels(self, y, y_hat):
        self.labels.append(y)
        self.preds.append(y_hat)

    def display_batch_imgs(self, x, y_hat_np, y_np, title="val images"):
        mask_list = []
        for original_image, true_mask, prediction_mask in zip(x, y_np, y_hat_np):
            mask_list.append(
                wandb.Image(
                    original_image.cpu(),
                    masks={
                        "prediction": {
                            "mask_data": np.argmax(prediction_mask, axis=0),
                            "class_labels": {0: "background", 1: "foreground"},
                        },
                        "ground truth": {
                            "mask_data": np.argmax(true_mask, axis=0),
                            "class_labels": {0: "background", 1: "foreground"},
                        },
                    },
                )
            )
        self.logger.experiment.log({title: mask_list})


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


loaders = get_palm_loaders(**loader_param)
tl, vl, ttl = loaders

bce_fn = torch.nn.BCEWithLogitsLoss()
dice_fn = DiceLoss()


def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred, y_true)
    return bce_weight * bce + dice_weight * dice


use_sch = set_scheduler != "none"
total_steps = epochs * len(tl) if use_sch else 0
model = (
    Model(
        n_classes,
        loss_fct=loss_fn,
        base_model=basemodel,
        lr=lr,
        total_steps=total_steps,
        pretrained=pretrained_imagenet,
        set_scheduler=set_scheduler,
        opt_method=optimizer,
        opt_param=optimizer_dict,
        seg_model_name=seg_model_name,
    )
    .cuda()
    .train()
)

logger = None
if WANDB:
    logger = WandbLogger(project="PALM_myopia_segmentation")
    params = {
        "epochs": epochs,
        "train_pct": train_pct,
        "lr": lr,
        "scheduler": set_scheduler,
        "base_model": basemodel.__name__,
        "img_size": tl.dataset[0][0].shape[-1],
        "bs": tl.batch_size,
        "accumulate_grad_batches": accumulate_grad_batches,
        "seg_model_name": seg_model_name,
    }
    logger.log_hyperparams(params)

trainer = pl.Trainer(
    gpus=1,
    deterministic=True,
    max_epochs=epochs,
    logger=logger if WANDB else True,
    accumulate_grad_batches=accumulate_grad_batches,
    callbacks=[
        ModelCheckpoint(
            monitor="valid_loss",
            filename="model-{epoch:02d}-{valid_dice:.2f}",
            save_top_k=1,
        ),
    ],
)
trainer.validate(model, dataloaders=vl)
trainer.fit(model, tl, vl)

result = trainer.test(dataloaders=ttl, ckpt_path="best")
