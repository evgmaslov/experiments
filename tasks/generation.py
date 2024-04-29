from .base import TaskInput, TaskOutput
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class GenerationTaskInput(TaskInput):
    noise: Optional[torch.Tensor]

@dataclass  
class GenerationTaskOutput(TaskOutput):
    sample: torch.Tensor