import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lovasz_loss import lovasz_softmax


class lovasz_softmax_with_logit(nn.Module):
    def __init__(self, ignore=None):
        super(lovasz_softmax_with_logit, self).__init__()
        self.ignore = ignore
    def forward(self, probas, labels, classes='present', per_image=False):
        probas_s = F.softmax(probas, dim=1)
        return lovasz_softmax(probas_s, labels, ignore=self.ignore)
