from datasets import Dataset
from typing import Dict

class Converter():
    def __call__(self, *args, **kwargs) -> Dataset:
        pass

class Printer():
    def __call__(self, data: Dict):
        pass