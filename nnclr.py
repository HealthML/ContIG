import os

import lightly.loss as loss
import lightly.models as models
import pytorch_lightning as pl
import torch
import torchvision
from PIL import ImageFile
from lightly.models.modules.heads import ProjectionHead
from lightly.models.modules.nn_memory_bank import NNMemoryBankModule
from torch import nn

from data.data_ukb import get_imaging_pretraining_data

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ImageFile.LOAD_TRUNCATED_IMAGES = True

max_epochs = 100
IMG_SIZE = 448
PROJECTION_DIM = 128
BATCH_SIZE = 32
ACCUMULATE_GRAD_BATCHES = 2
LR = 1e-3
WEIGHT_DECAY = 1e-6
TEMPERATURE = 0.1
MEMORY_BANK_SIZE = 2 ** 16


class NNCLRModel(pl.LightningModule):
    def __init__(self, num_ftrs=2048):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet50()
        # create a nnclr model based on ResNet
        self.resnet_nnclr = models.NNCLR(
            torch.nn.Sequential(*list(resnet.children())[:-1]),
            num_ftrs=num_ftrs,
            proj_hidden_dim=num_ftrs,
            pred_hidden_dim=num_ftrs,
            out_dim=PROJECTION_DIM,
        )
        self.resnet_nnclr.projection_mlp = ProjectionHead(
            [
                (
                    self.resnet_nnclr.num_ftrs,
                    self.resnet_nnclr.proj_hidden_dim,
                    nn.BatchNorm1d(self.resnet_nnclr.proj_hidden_dim),
                    nn.ReLU(inplace=True),
                ),
                (
                    self.resnet_nnclr.proj_hidden_dim,
                    self.resnet_nnclr.out_dim,
                    nn.BatchNorm1d(self.resnet_nnclr.out_dim),
                    None,
                ),
            ]
        )
        self.criterion = loss.NTXentLoss(
            temperature=TEMPERATURE, memory_bank_size=MEMORY_BANK_SIZE
        )
        self.nn_replacer = NNMemoryBankModule(size=MEMORY_BANK_SIZE)

    def forward(self, x):
        self.resnet_nnclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        (z0, p0), (z1, p1) = self.resnet_nnclr(x0, x1)

        z0 = self.nn_replacer(z0.detach(), update=False)
        z1 = self.nn_replacer(z1.detach(), update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        global training_set_len
        optim = torch.optim.Adam(
            self.resnet_nnclr.parameters(),
            LR,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=training_set_len, eta_min=0, last_epoch=-1
        )
        return [optim], [scheduler]


model = NNCLRModel()

print(model)

dataloader, _, _ = get_imaging_pretraining_data(
    num_workers=8,
    size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    train_pct=0.7,
    val_pct=0.1,
    tfms_settings="simclr",
)

training_set_len = len(dataloader)

trainer = pl.Trainer(
    max_epochs=max_epochs,
    gpus=1,
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
)
trainer.fit(model, dataloader)

print("Finished Training")
