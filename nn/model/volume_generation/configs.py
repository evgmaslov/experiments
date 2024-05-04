from ..base import ModelConfig
from ....methods.diffusion.schedulers import SchedulerConfig
from typing import Tuple
from dataclasses import dataclass

@dataclass
class DiffusionConfig(ModelConfig):
  sample_channels: int = 1
  sample_shape: Tuple = (32, 32, 32)
  model_channels: int = 64
  ch_mults: Tuple = (1, 2, 2, 4)
  attn_res: Tuple = (4)
  layers_per_block: int = 2
  dropout: float = 0
  conv_resample: bool = False
  num_heads: int = 1
  num_head_channels: int = -1
  use_scale_shift_norm: bool = False
  resblock_updown: bool = False
  split_qkv_before_heads: bool = False

  scheduler_config: SchedulerConfig
  loss: str = "MSELoss"

  def to_dict(self):
    d = super().to_dict()
    d["sample_channels"] = self.sample_channels
    d["sample_shape"] = self.sample_shape
    d["model_channels"] = self.model_channels
    d["ch_mults"] = self.ch_mults
    d["attn_res"] = self.attn_res
    d["layers_per_block"] = self.layers_per_block
    d["dropout"] = self.dropout
    d["conv_resample"] = self.conv_resample
    d["num_heads"] = self.num_heads
    d["num_head_channels"] = self.num_head_channels
    d["use_scale_shift_norm"] = self.use_scale_shift_norm
    d["resblock_updown"] = self.resblock_updown
    d["split_qkv_before_heads"] = self.split_qkv_before_heads
    d["scheduler_config"] = self.scheduler_config.to_dict()
    d["loss"] = self.loss
    return d

@dataclass
class SeisFusionConfig(DiffusionConfig):
  gama: float
  u: int = 10

  def to_dict(self):
    d = super().to_dict()
    d["gama"] = self.gama
    return d