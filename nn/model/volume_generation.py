from transformers.utils import ModelOutput
from nn.model import ModelConfig, Model
from nn.model.configs import DiffusionConfig

import torch
from torch import nn 
from diffusers import UNet3DConditionModel, DDPMScheduler
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
from typing import Dict

class DiffusersDDPM3D(Model):
  def __init__(self, config: DiffusionConfig):
    super().__init__(config)
    self.cross_attention_dim = 8
    self.sample_shape = config.sample_shape
    self.sample_channels = config.sample_channels
    self.layers_per_block = config.layers_per_block
    self.channels = config.channels
    self.eps_model = UNet3DConditionModel(sample_size = self.sample_shape, in_channels = self.sample_channels, out_channels = self.sample_channels,
                                 layers_per_block = self.layers_per_block, cross_attention_dim=self.cross_attention_dim, block_out_channels=self.channels)
    self.noise_scheduler = DDPMScheduler()
    self.loss = nn.MSELoss()
  def forward(self, x: Dict) -> ModelOutput:
    x0 = x["volume"]
    batch_size = x0.shape[0]
    t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0).to(x0.device)

    dummy_cond = torch.zeros(batch_size, x0.shape[1], self.cross_attention_dim).to(self.device)
    xt = self.noise_scheduler.add_noise(x0, noise, t)
    eps_theta = self.eps_model(xt, t, dummy_cond)[0]
    loss = self.loss(eps_theta, noise)

    output = ModelOutput()
    output["loss"] = loss
    return output
  def inference(self, x: Dict, print_output: bool = False) -> Dict:
    with torch.no_grad():
      x = torch.randn_like(x["volume"])
      dummy_cond = torch.zeros(x.shape[0], x.shape[2], self.cross_attention_dim).to(self.device)
      for t in tqdm(self.noise_scheduler.timesteps):
        time = x.new_full((x.shape[0],), t, dtype=torch.long)
        xt = self.eps_model(x, time, dummy_cond)[0]
        x = self.noise_scheduler.step(xt, t, x).prev_sample
      output = {
          "volume":x
      }
    if print_output:
      self.show_outputs(x)
    return output
  def show_outputs(self, x):
      x = x[:,0,:,:,:].squeeze(1)
      x = abs(x + abs(x.min()))
      x = (x/x.max()*255).type("torch.LongTensor")

      sample = x[0].detach().cpu().numpy()
      fig = plt.figure()
      for i in range(x.shape[0]):
        tess = x[i].detach().cpu().numpy()[:,:,:,np.newaxis].repeat(3, axis=3)/255
        color = np.concatenate([tess, np.ones((*self.sample_shape, 1))], axis=3)
        ax = fig.add_subplot(projection='3d')
        ax.voxels(sample, facecolors=color)
      plt.show()