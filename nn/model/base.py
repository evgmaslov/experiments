from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput 
from dataclasses import dataclass, fields
from typing import Dict

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