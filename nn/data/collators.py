from typing import List
from nn.data import CollatorConfig
from .base import Collator
import numpy as np
import torch

class TesrCollator(Collator):
    def __init__(self, config: CollatorConfig):
        super().__init__(config)
    def __call__(self, batch: List):
        new_batch = {
            "volume":[]
        }
        for element in batch:
            tess = self.read_tesr(element["volume"])[np.newaxis,:,:,:]
            new_batch["volume"].append(tess.tolist())
        new_batch["volume"] = torch.from_numpy(np.concatenate(new_batch["volume"], axis=0))
        return new_batch
    def read_tesr(self, lines):
      voxels = []
      n_cells = 0
      size = 0
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