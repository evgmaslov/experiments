from datasets import Dataset
from typing import Dict, List
from . import CollatorConfig

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