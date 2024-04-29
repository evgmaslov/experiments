from .base import Converter
from typing import Optional, List
import os
import torch
from datasets import Features, Array5D, Dataset
import numpy as np
from tqdm.notebook import tqdm

class TesrConverter(Converter):
    def __call__(self, path: str, names: List[str] = None, dims: int = 3) -> Dataset:
        names = names if names != None else os.listdir(path)
        data = []
        for name in tqdm(names, desc="Converting data"):
            full_path = f"{path}/{name}"
            with open(full_path, "r") as f:
                lines = f.readlines()
            data.append(lines.tolist())
        data = np.concatenate(data, axis=0)
        ds = Dataset.from_dict({"volume": data}).with_format("torch")
        return ds

