from nn.model import ModelConfig
from typing import Tuple
from dataclasses import dataclass

@dataclass
class DiffusionConfig(ModelConfig):
  sample_channels: int = 1
  sample_shape: Tuple = (32, 32, 32)
  layers_per_block: int = 2
  channels: Tuple = (320, 640, 1280, 1280)