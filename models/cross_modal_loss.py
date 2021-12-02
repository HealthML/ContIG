import torch
import torch.nn.functional as F

LARGE_NUM = 1e9


class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, alpha_weight):
        """Compute loss for model.
        temperature: a `floating` number for temperature scaling.
        weights: a weighting number or vector.
        """
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs, norm=True):
        temperature = self.temperature
        alpha = self.alpha_weight

        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(
            torch.arange(start=0, end=batch_size, dtype=torch.int64),
            num_classes=batch_size,
        ).float()
        labels = labels.to(self.device)

        # Different from Image-Image contrastive learning
        # In the case of Image-Gen contrastive learning we do not compute the intra-modal similarity
        # masks = F.one_hot(
        #     torch.arange(start=0, end=batch_size, dtype=torch.int64),
        #     num_classes=batch_size,
        # )
        # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
        # logits_aa = logits_aa - masks * LARGE_NUM
        # logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_bb = logits_bb - masks * LARGE_NUM

        logits_ab = (
            torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        )
        logits_ba = (
            torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature
        )

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return alpha * loss_a + (1 - alpha) * loss_b
