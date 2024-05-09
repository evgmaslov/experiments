from .base import Converter
from typing import Optional, List, Dict
import os
import torch
from datasets import Features, Array5D, Dataset, load_dataset
import numpy as np
from tqdm.notebook import tqdm
from .base import ConverterConfig
from dataclasses import dataclass
from . import STRING_TO_CONVERTER

class TesrConverter(Converter):
    def __call__(self, data: Dict) -> Dataset:
        path = data["path"]
        names = data.get("names", None)
        dims = data.get("dims", 3)
        names = names if names != None else os.listdir(path)
        data = []
        for name in tqdm(names, desc="Converting data"):
            full_path = f"{path}/{name}"
            with open(full_path, "r") as f:
                lines = f.readlines()
            data.append("".join(lines))
        ds = Dataset.from_dict({"volume": data})
        return ds

class LoadDatasetConverter(Converter):
    def __call__(self, data: Dict) -> Dataset:
        dataset = load_dataset(data["path"])
        return dataset

@dataclass
class ComposeConverterConfig(ConverterConfig):
    child_configs: List[ConverterConfig]

    def to_dict(self):
        d = super().to_dict()
        d["child_configs"] = [c.to_dict() for c in self.child_configs]
        return d

class ComposeConverter(Converter):
    def __init__(self, config: ComposeConverterConfig):
        super().__init__(config)
        self.converters = []
        for config in config.child_configs:
            converter_type = STRING_TO_CONVERTER.get(config.type, None)
            assert converter_type != None, f"Converter type {config.type} isn't registered."
            converter = converter_type(config)
            self.converters.append(converter)
    def __call__(self, data: torch.Dict) -> Dataset:
        dataset = None
        for converter in self.converters:
            local_dataset = converter(data)
            if dataset == None:
                dataset = local_dataset
            else:
                1
        return dataset