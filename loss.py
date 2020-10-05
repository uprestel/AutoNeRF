import numpy as np
import torch.nn as nn
import torch

from AutoNeRF.util import retrieve


def nll(sample):
    return 0.5*torch.sum(torch.pow(sample, 2), dim=[1,2,3])


class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, sample, logdet, global_step):
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss + nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        log = {"images": {},
               "scalars": {
                    "loss": loss, "reference_nll_loss": reference_nll_loss,
                    "nlogdet_loss": nlogdet_loss, "nll_loss": nll_loss,
               }}
        def train_op():
            pass
        return loss, log, train_op
