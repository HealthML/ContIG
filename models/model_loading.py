import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from models.cross_modal_model import ModelCLR

EMBEDDING_SIZE = 128


class LoaderModel(pl.LightningModule):
    def __init__(self, input_features_dims, h1=None, h2=None):
        super().__init__()
        if h1 is None:
            gname = None
        elif h1 == 2048 and h2 is None:
            gname = "H1_2048"
        elif h1 == 2048 and h2 == 2048:
            gname = "H12_2048"
        self.model = ModelCLR(
            gen_input_feats=input_features_dims,
            out_dim=EMBEDDING_SIZE,
            hidden1_size=h1,
            hidden2_size=h2,
            genetics_model_name=gname,
        )

    def init(self, path):
        print(self.load_state_dict(torch.load(path, map_location="cpu")["state_dict"]))


class LoaderModelCrossModal(pl.LightningModule):
    MODALITIES = ["raw_snps", "risk_scores", "burden_scores"]

    def __init__(self, input_feature_dims, h1=None, h2=None):
        super().__init__()
        self.models = nn.ModuleList()
        shared_img_encoder = None
        if h1 is None:
            gname = None
        elif h1 == 2048 and h2 is None:
            gname = "H1_2048"
        elif h1 == 2048 and h2 == 2048:
            gname = "H12_2048"
        for modality in self.MODALITIES:
            feat_dim = input_feature_dims[
                {
                    "raw_snps": "gen",
                    "risk_scores": "pgs",
                    "burden_scores": "burdens",
                }[modality]
            ]
            if len(self.models) > 0:
                shared_img_encoder = self.models[0].imaging_model
            self.models.append(
                ModelCLR(
                    gen_input_feats=feat_dim,
                    shared_img_encoder=shared_img_encoder,
                    out_dim=EMBEDDING_SIZE,
                    hidden1_size=h1,
                    hidden2_size=h2,
                    genetics_model_name=gname,
                )
            )

    def init(self, path):
        print(self.load_state_dict(torch.load(path, map_location="cpu")["state_dict"]))


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


class GeneralModel(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path=None,
        base_model=torchvision.models.resnet50,
        pretrained=True,
        device="cpu",
    ):
        super().__init__()
        if checkpoint_path is None:
            self.model = base_model(pretrained=pretrained)
        else:
            self.model = base_model()
            state_dict = torch.load(checkpoint_path, map_location=device)
            if (
                "simclr" in checkpoint_path
                or "byol" in checkpoint_path
                or "barlow" in checkpoint_path
                or "simsiam" in checkpoint_path
                or "nnclr" in checkpoint_path
            ):
                load_from_state_dict_img_only(self.model, state_dict["state_dict"])
            else:
                if "state_dict" in state_dict:
                    load_from_state_dict_gen_img(self.model, state_dict["state_dict"])
                else:
                    load_from_state_dict_gen_img(self.model, state_dict)

        self.model.fc = nn.Identity()  # nn.Linear(self.model.fc.in_features, 5)
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)
