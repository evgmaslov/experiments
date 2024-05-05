from typing import List, Tuple
from .base import Collator, CollatorConfig
import numpy as np
import torch
from dataclasses import dataclass
import random

class TesrCollator(Collator):
    def __init__(self, config: CollatorConfig):
        super().__init__(config)
    def __call__(self, batch: List):
        new_batch = {
            "volume":[]
        }
        for element in batch:
            lines = element["volume"].split("\n")
            tess = self.read_tesr(lines)[np.newaxis, np.newaxis,:,:,:]
            new_batch["volume"].append(tess.tolist())
        new_batch["volume"] = torch.from_numpy(np.concatenate(new_batch["volume"], axis=0)).type("torch.FloatTensor")
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

@dataclass
class TesrCollatorWithConditionConfig(CollatorConfig):
    condition_views: list
    condition_sampling_weights: list

class TesrCollatorWithCondition(TesrCollator):
    def __init__(self, config: CollatorConfig):
        super().__init__(config)
    def __call__(self, batch: List):
        batch = super().__call__(batch)
        conditions = random.choices(self.config.condition_views, weights=self.config.condition_sampling_weights, k=batch["volume"].shape[0])
        masks = None
        for c in conditions:
            mask = self.get_mask(c, batch["volume"].shape[2:])[None, None, :, :, :]
            if masks == None:
                masks = mask
            else:
                masks = torch.cat([masks, mask], dim=0)
        batch["mask"] = masks
        return batch
    def get_mask(self, condition: Tuple, shape) -> Tuple:
        mask = torch.zeros(shape)
        for ind, i in enumerate(condition):
            if i == 1:
                if ind == 0:
                  mask[0,:,:] = 1
                elif ind == 1:
                  mask[:,0,:] = 1
                elif ind == 2:
                  mask[:,:,0] = 1
            elif i == 2:
                if ind == 0:
                  mask[-1,:,:] = 1
                elif ind == 1:
                  mask[:,-1,:] = 1
                elif ind == 2:
                  mask[:,:,-1] = 1
        return mask