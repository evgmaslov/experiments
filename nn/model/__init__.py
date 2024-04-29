from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput 
from tasks import TaskInput
from dataclasses import dataclass, fields
from typing import Callable, Dict, Any
from volume_generation import DiffusersDDPM3D

@dataclass
class ModelConfig(PretrainedConfig):
    name: str
    type: str
    path: str

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

class Model(PreTrainedModel):
    def __init__(self, config: ModelConfig):
        super().__init__(self, config=config)
        self.config = config
    def forward(self, x: Dict) -> ModelOutput:
        pass
    def inference(self, x: Dict, print_output: bool = False) -> Dict:
        pass

STRING_TO_MODEL = {
    "Model":Model,
    "DiffusersDDPM3D":DiffusersDDPM3D,
}