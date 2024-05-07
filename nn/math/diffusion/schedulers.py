from typing import Tuple, Any, List, Dict
import numpy  as np
from dataclasses import dataclass, fields
import diffusers
import inspect
from .schedules import NoiseSchedule
from .utils import Respacing
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput

@dataclass
class SchedulerConfig:
    type: str
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

class DiffusersScheduler():
    def __init__(self, config: SchedulerConfig):
        diffusers_type = getattr(diffusers, config.type, None)
        assert diffusers_type != None, f"There are no {config.type} in Diffusers, please check out the docs."
        params = inspect.signature(diffusers_type).parameters
        config_params = {field.name: getattr(config, field.name) for field in fields(config) if field.init}
        final_params = {}
        for arg, value in params.items():
            if arg in config_params:
                final_params[arg] = config_params[arg]
        self.scheduler = diffusers_type(**final_params)
        self.config = self.scheduler.config
        self.timesteps = self.scheduler.timesteps
        self.betas = self.scheduler.betas
    def step(self, *args, **kwargs):
        return self.scheduler.step(*args, **kwargs)
    def add_noise(self, *args, **kwargs):
        return self.scheduler.add_noise(*args, **kwargs)
    def add_noise_step(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        betas = self.betas.to(t.device)
        betas = betas[t]
        img_in_est = torch.sqrt(1 - betas) * x + torch.sqrt(betas) * noise
        return img_in_est

@dataclass  
class SeisFusionSchedulerConfig(SchedulerConfig):
    beta_rescaling: bool = False
    respacing: List = None
    clip_denoised: bool = True
    rescale_timesteps: bool = False
    mean_type: str = "epsilon"
    variance_type: str = "learned"
    loss_type: str = "mse"

class SeisFusionScheduler():
    def __init__(self, config: SeisFusionSchedulerConfig):
        mean_types = ["epsilon", "fixed"]
        variance_types = ["learned", "learned_large", "fixed", "fixed_large"]
        assert config.mean_type in mean_types, f"Mean type should be in {', '.join(mean_types)}"
        assert config.variance_type in variance_types, f"Variance type should be in {', '.join(variance_types)}"

        self.config = config

        if config.respacing != None:
            self.respacing = Respacing(num_base_timesteps=config.num_train_timesteps, timestep_respacing=config.respacing)
        else:
            self.respacing = None

        self.schedule = NoiseSchedule(num_timesteps=config.num_train_timesteps, beta_schedule_type=config.beta_schedule,
                                      beta_start_end_values=(config.beta_start, config.beta_end), beta_rescale=config.beta_rescaling, 
                                      respacing=self.respacing)
        self.timesteps = self.schedule.timesteps
        
    def step(self, eps: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor, return_mean_and_var: bool = False):
        means = self.get_step_mean(eps, x_t, t)
        variances = self.get_step_variance(eps, x_t, t)

        noise = torch.randn_like(x_t)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when t == 0

        sample = means["mean"] + nonzero_mask * torch.exp(0.5 * variances["log_variance"]) * noise
        output = DDPMSchedulerOutput(prev_sample=sample, pred_original_sample=means["pred_xstart"])
        if return_mean_and_var:
            return output, means, variances
        else:
            return output
    
    def get_step_mean(self, model_output: torch.Tensor, model_input: torch.Tensor, t: torch.Tensor) -> Dict:
        step_values = self.schedule.get_step_values(t, model_input.shape)
        if self.config.variance_type in ("learned", "learned_range"):
            model_output, model_var_values = torch.chunk(model_output, 2, dim=1)

        pred_xstart = None
        if self.config.mean_type == "epsilon":
            pred_xstart = step_values["sqrt_recip_alphas_cumprod"]*model_input + step_values["sqrt_recipm1_alphas_cumprod"]*model_output
        elif self.config.mean_type == "previous_x":
            pred_xstart = 1/step_values["posterior_mean_coef1"]*model_output + step_values["posterior_mean_coef2"]*model_input

        if self.config.clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        mean = step_values["posterior_mean_coef1"]*pred_xstart + step_values["posterior_mean_coef2"]*model_input
        return {"mean":mean, "pred_xstart":pred_xstart}
    
    def get_step_variance(self, model_output: torch.Tensor, model_input: torch.Tensor, t: torch.Tensor) -> Dict:
        step_values = self.schedule.get_step_values(t, model_input.shape)

        c = model_output.shape[1]

        model_variance = None
        model_log_variance = None
        if self.config.variance_type == "learned":
            model_output, model_var_values = torch.chunk(model_output, c, dim=1)
            model_log_variance = model_var_values
            model_variance = torch.exp(model_var_values)
        elif self.config.variance_type == "learned_range":
            model_output, model_var_values = torch.chunk(model_output, c, dim=1)
            min_log = step_values["posterior_log_variance_clipped"]
            max_log = torch.log(step_values["betas"])

            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        elif self.config.variance_type == "fixed":
            model_variance = step_values["posterior_variance"]
            model_log_variance = step_values["posterior_log_variance_clipped"]
        elif self.config.variance_type == "fixed_large":
            step_values_1 = self.schedule.get_step_values(torch.ones_like(t).to(model_input.device), model_input.shape)
            first = t == 0
            model_variance = step_values["betas"]
            model_variance[first] = step_values["posterior_variance"]
            model_log_variance = torch.log(step_values["betas"])
            model_log_variance[first] = step_values["posterior_variance"]
        return {"variance":model_variance, "log_variance":model_log_variance}
    
    def transform_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        if self.config.rescale_timesteps:
            t = self.scale_timesteps(t)
        if self.respacing != None:
            t = self.respacing.respaced_timesteps_to_base(t)
        return t
    
    def scale_timesteps(self, t):
        return t.float() * (1000.0 / self.schedule.num_timesteps)
    
    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        step_values = self.schedule.get_step_values(t, x.shape)
        alpha_cumprod = step_values["alphas_cumprod"]

        gt_weight = torch.sqrt(alpha_cumprod)
        gt_part = gt_weight * x

        noise_weight = torch.sqrt((1 - alpha_cumprod))
        noise_part = noise_weight * torch.randn_like(x)

        weighed_gt = gt_part + noise_part
        return weighed_gt
    
    def add_noise_step(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        step_values = self.schedule.get_step_values(t, x.shape)
        beta = step_values["betas"]

        img_in_est = torch.sqrt(1 - beta) * x + \
                     torch.sqrt(beta) * noise

        return img_in_est
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        step_values = self.schedule.get_step_values(t, x_start.shape)
        posterior_mean = step_values["posterior_mean_coef1"] * x_start + step_values["posterior_mean_coef2"] * x_t
        
        posterior_variance = step_values["posterior_variance"]
        posterior_log_variance_clipped = step_values["posterior_log_variance_clipped"]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped