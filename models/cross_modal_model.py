import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.genetics_model import DietNetworkBasic

GENETICS_EMBEDDING_SIZE = 2048
IMG_EMBEDDING_SIZE = 2048


class ModelCLR(nn.Module):
    def __init__(
        self,
        gen_input_feats,
        shared_img_encoder=None,
        hidden1_size=2048,
        hidden2_size=2048,
        genetics_model_name=None,
        out_dim=128,
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
        else:
            gen_embed_size = GENETICS_EMBEDDING_SIZE
        # projection MLP for genetics model
        self.genetics_l1 = nn.Linear(gen_embed_size, GENETICS_EMBEDDING_SIZE)
        self.genetics_l2 = nn.Linear(GENETICS_EMBEDDING_SIZE, out_dim)

        # # Imaging
        if shared_img_encoder is not None:
            self.imaging_model = shared_img_encoder
        else:
            resnet = models.resnet50(pretrained=False)
            self.imaging_model = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP for imaging Model
        self.imaging_l1 = nn.Linear(IMG_EMBEDDING_SIZE, IMG_EMBEDDING_SIZE)
        self.imaging_l2 = nn.Linear(IMG_EMBEDDING_SIZE, out_dim)

    @staticmethod
    def init_genetics_model(
        gen_model_name,
        gen_input_feats,
        hidden1_size=2048,
        hidden2_size=2048,
    ):
        """
        genetics_model_name: can be "None", "H1_2048", "H12_2048"
        """
        if gen_model_name == "H1_2048" or gen_model_name == "H12_2048":
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
