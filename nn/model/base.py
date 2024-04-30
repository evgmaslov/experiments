from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput 
from dataclasses import dataclass, fields
from typing import Dict
from torch import nn

@dataclass
class ModelConfig(PretrainedConfig):
    name: str = ""
    type: str = ""
    path: str = ""

    def to_dict(self):
        d = super().to_dict()
        d["name"] = self.name
        d["type"] = self.type
        d["path"] = self.path
        return d

class Model(PreTrainedModel):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
    def forward(self, x: Dict) -> ModelOutput:
        pass
    def inference(self, x: Dict, print_output: bool = False) -> Dict:
        pass