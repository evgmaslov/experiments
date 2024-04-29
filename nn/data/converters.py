from . import Converter
from typing import Optional, List
import os
import torch
from datasets import Features, Array5D, Dataset
import numpy as np

class TesrConverter(Converter):
    def __call__(self, path: str, names: List[str] = None, dims: int = 3) -> Dataset:
        names = names if names != None else os.listdir(path)
        data = []
        for name in names:
            path = f"{path}/{name}"
            tess = self.read_tesr(path)[np.newaxis,:,:,:]
            if dims == 2:
                tess = tess[:,0,:,:].squeeze(1)
            data.append(tess.tolist())
        data = np.concatenate(data, axis=0)
        features = Features({"volume": Array5D(shape=data.shape, dtype='float32')})
        ds = Dataset.from_dict({"volume": data}, features=features).with_format("torch")
        return ds

    def read_tesr(self, path):
      voxels = []
      n_cells = 0
      size = 0
      with open(path, "r") as f:
          lines = f.readlines()
      read_mode = "base"
      for s in lines:
          if s.find("**cell") != -1:
              read_mode = "cell"
              continue
          elif s.find("**data") != -1:
              read_mode = "pre_data"
              continue
          elif s.find("***end") != -1:
              break
          elif s.find("**general") != -1:
              read_mode = "pre_general"
              continue
          if read_mode == "pre_data":
              read_mode = "data"
              continue
          elif read_mode == "pre_general":
              read_mode = "general"
              continue
          elif read_mode == "cell":
              n_cells = int(s.strip())
              read_mode = "base"
          elif read_mode == "data":
              new_voxels = [int(i) for i in s.strip().split(" ")]
              voxels.extend(new_voxels)
          elif read_mode == "general":
              size = [int(i) for i in s.strip().split(" ")][0]
              read_mode = "base"
      tess = np.zeros((size, size, size))
      counter = 0
      for i in range(size):
        for j in range(size):
          for k in range(size):
            tess[k, j, i] = voxels[counter]/n_cells
            counter += 1
      return tess
