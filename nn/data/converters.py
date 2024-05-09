from .base import Converter
from typing import Optional, List, Dict
import os
import torch
from datasets import Features, Array5D, Dataset, load_dataset
import numpy as np
from tqdm.notebook import tqdm
from .base import ConverterConfig
from dataclasses import dataclass

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