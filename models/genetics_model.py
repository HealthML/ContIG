import torch
import torch.nn as nn


class DietNetworkBasic(nn.Module):
    def __init__(
        self,
        n_feats,  # this can change depending on what kind of feature dimension reduction was done
        n_hidden1_u,
        n_hidden2_u=None,
        eps=1e-05,
    ):
        super(DietNetworkBasic, self).__init__()

        # 1st hidden layer
        self.hidden_1 = nn.Linear(n_feats, n_hidden1_u)
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden1_u, eps=eps)

        # 2nd hidden layer
        self.hidden_2 = None
        if n_hidden2_u is not None:
            self.hidden_2 = nn.Linear(n_hidden1_u, n_hidden2_u)
            self.bn2 = nn.BatchNorm1d(num_features=n_hidden2_u, eps=eps)

    def forward(self, x):
        z1 = self.hidden_1(x)
        a1 = torch.relu(z1)
        a1 = self.bn1(a1)
        out = a1

        if self.hidden_2 is not None:
            z2 = self.hidden_2(a1)
            a2 = torch.relu(z2)
            a2 = self.bn2(a2)
            out = a2

        return out
