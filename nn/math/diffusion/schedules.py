import numpy as np
from typing import Tuple, List, Dict
import math
import torch
from .utils import Respacing

class BetaSchedule():
    types = ("linear", "cosine")

    def get(self, type: str = "linear", num_timesteps: int = 1000, start_end_values: Tuple = (0.0001, 0.02),  rescale: bool = False) -> List:
        if type == "linear":
            return self.linear(num_timesteps, start_end_values, rescale)
        elif type == "cosine":
            return self.cosine(num_timesteps, start_end_values)
        
    def linear(self, num_timesteps, start_end_values, rescale) -> List:
        if rescale:
            scale = 1000 / num_timesteps
        else:
            scale = 1

        beta_start = scale * start_end_values[0]
        beta_end = scale * start_end_values[1]
        return np.linspace(
            beta_start, beta_end, num_timesteps, dtype=np.float64
        ).tolist()
    
    def cosine(self, num_timesteps, start_end_values) -> List:
        alpha_fn = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - alpha_fn(t2) / alpha_fn(t1), start_end_values[1]))
        return betas

class NoiseSchedule():
    def __init__(self, num_timesteps: int = 1000, beta_schedule_type: str = "cosine", beta_start_end_values: Tuple = (0.0001, 0.02), beta_rescale: bool = False,
                 respacing: Respacing = None):
        betas = BetaSchedule().get(beta_schedule_type, num_timesteps, beta_start_end_values, beta_rescale)
        self.num_timesteps = num_timesteps
        self.timesteps = range(num_timesteps)[::-1]

        if respacing != None:
            betas = respacing.respace_betas(betas)
            self.num_timesteps = len(betas)
            self.respaced_timesteps = respacing.respaced_timesteps[::-1]
            self.timesteps = range(len(self.respaced_timesteps))[::-1]

        self.betas = np.array(betas, dtype=np.float64)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
    def get_step_values(self, t: torch.Tensor, shape: None) -> Dict:
        values = {
            "betas":torch.from_numpy(self.betas).to(t.device)[t],
            "alphas":torch.from_numpy(self.alphas).to(t.device)[t],
            "alphas_cumprod":torch.from_numpy(self.alphas_cumprod).to(t.device)[t],
            "sqrt_alphas_cumprod":torch.from_numpy(self.sqrt_alphas_cumprod).to(t.device)[t],
            "sqrt_one_minus_alphas_cumprod":torch.from_numpy(self.sqrt_one_minus_alphas_cumprod).to(t.device)[t],
            "log_one_minus_alphas_cumprod":torch.from_numpy(self.log_one_minus_alphas_cumprod).to(t.device)[t],
            "sqrt_recip_alphas_cumprod":torch.from_numpy(self.sqrt_recip_alphas_cumprod).to(t.device)[t],
            "sqrt_recipm1_alphas_cumprod":torch.from_numpy(self.sqrt_recipm1_alphas_cumprod).to(t.device)[t],
            "posterior_variance":torch.from_numpy(self.posterior_variance).to(t.device)[t],
            "posterior_log_variance_clipped":torch.from_numpy(self.posterior_log_variance_clipped).to(t.device)[t],
            "posterior_mean_coef1":torch.from_numpy(self.posterior_mean_coef1).to(t.device)[t],
            "posterior_mean_coef2":torch.from_numpy(self.posterior_mean_coef2).to(t.device)[t],
        }
        if shape == None:
            return values
        else:
            for key in values.keys():
                value = values[key]
                while len(value.shape) < len(shape):
                    value = value[..., None]
                values[key] = value
            return values