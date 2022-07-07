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

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.genetics_model import DietNetworkBasic

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True

pl.seed_everything(42)

IMG_SIZE = 448
BATCH_SIZE = 128
ACCUMULATE_GRAD_BATCHES = 1
LR = 1e-3
WEIGHT_DECAY = 1e-6
GEN_INPUT_L1_PENALTY = 1e-6
TEMPERATURE = 0.1
ALPHA_WEIGHT = 0.75
EPOCHS = 20
EVAL_EVERY_N_EPOCHS = 5
RESNET_MODEL_NAME = "resnet18"  # can be "resnet18" or "resnet50"
GENETICS_MODEL_NAME = "H1" # can be "H1" or "H12" or None

MIN_GEN_BURDEN_CARRIER = 10

GENETICS_HIDDEN_SIZE = 64
GENETICS_EMBEDDING_SIZE = 32
IMG_EMBEDDING_SIZE = 512
CM_EMBEDDING_SIZE = 32


COV_NOISE_SIZE = 5
# 'raw_snps' or 'risk_scores' or 'burden_scores' or 'covs'
# or it can be a list of combinations of these modalities ["raw_snps", "risk_scores", "burden_scores"]
GENETICS_MODALITY = "burden_scores"
AGGREGATE_MODALITIES = "inner"  # "inner" or "outer"

device = "cuda" if torch.cuda.is_available() else "cpu"

WORKERS = 8

H1 = None
H2 = None
if GENETICS_MODEL_NAME == "H1":
    H1 = GENETICS_HIDDEN_SIZE
elif GENETICS_MODEL_NAME == "H12":
    H1 = GENETICS_HIDDEN_SIZE
    H2 = GENETICS_HIDDEN_SIZE

# overwrite implementation of the class in models-module, in order to make the layer sizes configurable
class ModelCLR(nn.Module):
    def __init__(
        self,
        gen_input_feats,
        shared_img_encoder=None,
        hidden1_size=2048,
        hidden2_size=2048,
        genetics_model_name=None,
        out_dim=128,
        use_pretrained_img_encoder=False
    ):
        super(ModelCLR, self).__init__()
        # # Genetics
        self.genetics_model = self.init_genetics_model(
            genetics_model_name,
            gen_input_feats,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
        )
        if genetics_model_name is None:
            gen_embed_size = gen_input_feats
        elif genetics_model_name == 'H1':
            gen_embed_size = hidden1_size
        else:
            gen_embed_size = hidden2_size
            
        # projection MLP for genetics model
        self.genetics_l1 = nn.Linear(gen_embed_size, GENETICS_EMBEDDING_SIZE)
        self.genetics_l2 = nn.Linear(GENETICS_EMBEDDING_SIZE, out_dim)

        # # Imaging
        img_embed_size = IMG_EMBEDDING_SIZE # the size of the input to the image MLP projection head
        if shared_img_encoder is not None:
            self.imaging_model = shared_img_encoder
        else:
            if RESNET_MODEL_NAME == 'resnet18':
                resnet = models.resnet18(pretrained=use_pretrained_img_encoder)
                img_embed_size = 512
            elif RESNET_MODEL_NAME == 'resnet50':
                resnet = models.resnet50(pretrained=use_pretrained_img_encoder)
                img_embed_size = 2048
            else:
                raise NotImplementedError(f'model "{RESNET_MODEL}" not implemented')
            self.imaging_model = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP for imaging Model
        self.imaging_l1 = nn.Linear(img_embed_size, IMG_EMBEDDING_SIZE)
        self.imaging_l2 = nn.Linear(IMG_EMBEDDING_SIZE, out_dim)

    @staticmethod
    def init_genetics_model(
        gen_model_name,
        gen_input_feats,
        hidden1_size=2048,
        hidden2_size=2048,
    ):
        """
        genetics_model_name: can be "None", "H1", "H12"
        """
        if gen_model_name == "H1" or gen_model_name == "H12":
            model = DietNetworkBasic(
                n_feats=gen_input_feats,
                n_hidden1_u=hidden1_size,
                n_hidden2_u=hidden2_size,
            )
        else:
            model = nn.Identity()
        return model

    def image_encoder(self, xis):
        h = self.imaging_model(xis)
        h = h.squeeze()

        x = self.imaging_l1(h)
        x = F.relu(x)
        out_emb = self.imaging_l2(x)

        return out_emb

    def genetics_encoder(self, xjs):
        outputs = self.genetics_model(xjs)

        x = self.genetics_l1(outputs)
        x = F.relu(x)
        out_emb = self.genetics_l2(x)

        return out_emb

    def forward(self, xis, xjs):
        zis = self.image_encoder(xis)

        zjs = self.genetics_encoder(xjs)

        return zis, zjs  

    
class CrossModalModel(pl.LightningModule):
    def __init__(self, input_features_dims, l1_gen_input_penalty=None):
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
        
        if l1_gen_input_penalty is not None:
            assert not self.multimodal, 'l1_gen_input_penalty not yet supported for multi-modal models with more than one genetics modality.'
            self.l1_gen_input_penalty = l1_gen_input_penalty
        else:
            self.l1_gen_input_penalty = None
            
            
    def forward(self, x):
        if not self.multimodal:
            self.model(x)
        else:
            self.models[0](x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        
        penalty = None
        
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
            
            if self.l1_gen_input_penalty is not None:
                if isinstance(self.model.genetics_model, nn.Identity):
                    l1_norm = sum(w.abs().sum() for n, w in self.model.genetics_l1.named_parameters() if n == 'weight')
                else:
                    l1_norm = sum(w.abs().sum() for n, w in self.model.genetics_model.named_parameters() if n == 'hidden_1.weight')
                penalty = l1_norm * self.l1_gen_input_penalty
            
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
        
        if penalty is not None:
            self.log("l1_penalty", penalty)
            loss = loss + penalty
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
        
    def training_epoch_end(self, outs):
        # TODO: print out the l1 penalty, summarize how many weights are very close to 0.
        pass
    

        
# Dataloaders
loaders = None
input_features_sizes = None
if GENETICS_MODALITY == "raw_snps":
    loaders, input_features_sizes = get_genetics_imaging_data(
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
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
        burdens_zeros=MIN_GEN_BURDEN_CARRIER,
        num_workers=WORKERS,
        train_pct=0.7,
        val_pct=0.1,
    )
elif GENETICS_MODALITY == "risk_scores":
    loaders, input_features_sizes = get_pgs_imaging_data(
        normalize_pgs=True,
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        train_pct=0.7,
        val_pct=0.1,
    )
elif GENETICS_MODALITY == "covs":
    loaders, input_features_sizes = get_imaging_card_data(
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
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
        num_workers=WORKERS,
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

model = CrossModalModel(input_features_dims=input_features_sizes, l1_gen_input_penalty=GEN_INPUT_L1_PENALTY)


# use this to over-fit some batches 
# TODO: trainer = Trainer(overfit_batches=10) 


# TODO: implement early stopping callback when all weights go to -> 0.
# can be implemented with "stopping_threshold"


# TODO: add other callbacks:
#from pytorch_lightning.callbacks import Callback


#class MyPrintingCallback(Callback):
#    def on_train_start(self, trainer, pl_module):
#        print("Training is starting")
#
#    def on_train_end(self, trainer, pl_module):
#        print("Training is ending")


# from pytorch_lightning.loggers import WandbLogger
# 
# # instrument experiment with W&B
# wandb_logger = WandbLogger(project="MNIST", log_model="all")
# trainer = Trainer(logger=wandb_logger)
# 
# # log gradients and model topology
# wandb_logger.watch(model)

# TODO: check out pytorch pruning functionality

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