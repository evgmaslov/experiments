from datasets import Dataset
from typing import Dict, List
from dataclasses import dataclass, fields

@dataclass
class CollatorConfig:
    type: str

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

class Converter():
    def __call__(self, *args, **kwargs) -> Dataset:
        pass

class Collator():
    def __init__(self, config: CollatorConfig):
        self.config = config
    def __call__(self, batch: List):
        pass

class Printer():
    def __call__(self, data: Dict):
        pass