from typing import List
import numpy as np
import torch

class Respacing():
    def __init__(self, num_base_timesteps: int, timestep_respacing: List):
        self.num_base_timesteps = num_base_timesteps
        self.respaced_timesteps = self.get_new_timesteps(num_base_timesteps, timestep_respacing)

    def respace_betas(self, betas: List) -> List:
        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        new_betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in self.respaced_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        return new_betas
    
    def respaced_timesteps_to_base(self, t: torch.Tensor):
        map_tensor = torch.Tensor(self.respaced_timesteps).type(t.dtype).to(t.device)
        new_t = map_tensor[t]
        return new_t

    def get_new_timesteps(self, num_base_timesteps: int, timestep_respacing: List) -> List:
        size_per = num_base_timesteps // len(timestep_respacing)
        extra = num_base_timesteps % len(timestep_respacing)
        start_idx = 0
        all_steps = []

        if len(timestep_respacing) == 1 and timestep_respacing[0] > num_base_timesteps:
            steps = list(set(np.linspace(start=0, stop=num_base_timesteps, num=timestep_respacing[0])))
            steps.sort()
            return steps

        for i, section_count in enumerate(timestep_respacing):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(
                    f"cannot divide section of {size} steps into {section_count}"
                )
            if section_count <= 1:
                frac_stride = 1
            else:
                frac_stride = (size - 1) / (section_count - 1)
            cur_idx = 0.0
            taken_steps = []
            for _ in range(section_count):
                taken_steps.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            all_steps += taken_steps
            start_idx += size
        all_steps = list(set(all_steps))
        all_steps.sort()
        return all_steps