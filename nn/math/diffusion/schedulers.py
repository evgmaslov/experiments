from typing import Tuple
import numpy  as np
from dataclasses import dataclass, fields
import diffusers
import inspect

@dataclass
class SchedulerConfig:
    type: str
    num_train_timesteps: int = 1000

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