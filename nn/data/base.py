from datasets import Dataset
from typing import Dict, List
from dataclasses import dataclass, fields

@dataclass
class CollatorConfig:
    type: str

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

@dataclass
class ConverterConfig:
    type: str

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

class Converter():
    def __init__(self, config: ConverterConfig):
        self.config = config
    def __call__(self, data: Dict) -> Dataset:
        pass

class Collator():
    def __init__(self, config: CollatorConfig):
        self.config = config
    def __call__(self, batch: List):
        pass

class Printer():
    def __call__(self, data: Dict):
        pass

@dataclass
class DataConfig:
    name: str
    path: str
    converter_config: ConverterConfig
    collator_config: CollatorConfig
    split: float = 0.2
    use_size: float = 1.0
    

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d