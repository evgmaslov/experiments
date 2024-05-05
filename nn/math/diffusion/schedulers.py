from diffusers import SchedulerMixin, ConfigMixin
import torch
from typing import Tuple
import numpy  as np
from dataclasses import dataclass, fields

@dataclass
class SchedulerConfig(ConfigMixin):
    type: str

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d