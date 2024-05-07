from torch import nn 
import torch
from typing import Dict
from ...math.statistics.divergence import kl_divergence

class KLLoss():
    def __call__(self, pred_dist: Dict, true_dist: Dict, return_batch: bool = False) -> torch.Tensor:
        pred_mean, pred_var = torch.split(pred_dist, 2, dim=1)
        true_mean, true_var = torch.split(true_dist, 2, dim=1)

        kl = kl_divergence(true_mean, true_var, pred_mean, pred_var)
        if return_batch:
            return kl
        else:
            return kl.mean()
