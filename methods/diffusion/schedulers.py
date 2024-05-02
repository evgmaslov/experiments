from diffusers import SchedulerMixin, ConfigMixin
import torch
from typing import Tuple
import numpy  as np
from dataclasses import dataclass, fields

@dataclass
class SchedulerConfig(ConfigMixin):
    type: str
    num_timesteps: int
    beta_start: float = 0.0001
    beta_end: float = 0.02
    rescale_timesteps = False

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

class SeisFusionScheduler(SchedulerMixin):
    def __init__(self, config: SchedulerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_timesteps = config.num_timesteps

        betas = np.linspace(config.beta_start, config.beta_end, num=self.num_timesteps)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

    def step(self, model_output: torch.Tensor, timestep: int, x: torch.Tensor) -> Tuple:
        out = self.p_mean_variance(
            model_output,
            x,
            timestep,
        )
        return out["mean"]
    
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_0.shape) * x_0
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, x_0.shape)
            * noise
        )
    
    def sample_timesteps(self, num_samples):
        w = np.ones([self.num_timesteps])
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(num_samples,), p=p)
        indices = torch.from_numpy(indices_np).long()
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float()
        if self.rescale_timesteps:
            indices = indices.float() * (1000.0 / self.num_timesteps)
        return indices, weights
    
    def p_mean_variance(self, model_output, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        return {
            "mean": model_mean,
        }
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    