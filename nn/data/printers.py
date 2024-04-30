from typing import Dict
from .base import Printer
import matplotlib.pyplot as plt
import numpy as np

class VolumePrinter(Printer):
    def __call__(self, data: Dict):
        fig = plt.figure()
        volumes = data["volume"]
        sample_shape = volumes.shape[2:]
        for i in range(volumes.shape[0]):
            vol = volumes[i].permute(1, 2, 3, 0)
            sample = vol.detach().cpu().numpy()
            sample = (sample*255).astype("uint8")
            ax = fig.add_subplot(projection='3d')
            colors = sample.repeat(3, axis=3)/255
            colors = np.concatenate([colors, np.ones((*sample_shape, 1))], axis=3)
            ax.voxels(sample.squeeze(3), facecolors=colors)
        plt.show()