from nn.model.base import ModelConfig
from .. import Model
from ..blocks.diffusion import AttentionBlock, ResBlockWithTimestep, TimestepEmbedSequential, timestep_embedding
from ..blocks.vision import conv_nd, Downsample, Upsample
from ..utils.initialization import zero
from .configs import SeisFusionConfig
from ....methods.diffusion import STRING_TO_SCHEDULER
from ...train.loss import STRING_TO_LOSS

from typing import Dict
from transformers.utils import ModelOutput
from abc import abstractmethod
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        data_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        num_heads_upsample = num_heads

        self.data_size = data_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.updown_scale = (1, 2, 2) if dims == 3 else 2

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlockWithTimestep(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        zero_last_conv=True,
                        norm_groups=32,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            split_qkv_before_heads=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockWithTimestep(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            downsample=self.updown_scale,
                            zero_last_conv=True,
                            norm_groups=32,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, scale=self.updown_scale
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockWithTimestep(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                zero_last_conv=True,
                norm_groups=32,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                split_qkv_before_heads=use_new_attention_order,
            ),
            ResBlockWithTimestep(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                zero_last_conv=True,
                norm_groups=32,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockWithTimestep(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        zero_last_conv=True,
                        norm_groups=32,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            split_qkv_before_heads=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlockWithTimestep(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=self.updown_scale,
                            zero_last_conv=True,
                            norm_groups=32,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, scale=self.updown_scale)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h
        return self.out(h)

class condition_U(nn.Module):
    def __init__(self,
                 gama,
                 data_size,
                 in_channels,
                 model_channels,
                 out_channels,
                 num_res_blocks,
                 attention_resolutions,
                 dropout=0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=3,
                 num_heads=1,
                 num_head_channels=-1,
                 use_scale_shift_norm=False,
                 resblock_updown=False,
                 use_new_attention_order=False,
                 ):
        super().__init__()
        self.model1 = UNetModel(
            data_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_heads,
            num_head_channels,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
        )
        self.model2 = UNetModel(
            data_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_heads,
            num_head_channels,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
        )
        self.gama = gama

    def forward(self, x, t, condition):
        t1 = self.model1(x, t)
        t2 = self.model2(condition, t)

        return self.gama * t1 + (1 - self.gama) * t2

class SeisFusion(Model):
    def __init__(self, config: SeisFusionConfig):
        super().__init__(config)
        self.eps_model = condition_U(
            gama=config.gama,
            in_channels=config.sample_channels,
            model_channels=config.model_channels,
            out_channels=(1 if not config.learn_sigma else 2),
            num_res_blocks=config.layers_per_block,
            attention_resolutions=config.attn_res,
            dropout=config.dropout,
            channel_mult=config.ch_mults,
            num_heads=config.num_heads,
            num_head_channels=config.num_head_channels,
            use_scale_shift_norm=config.use_scale_shift_norm,
            resblock_updown=config.resblock_updown,
            use_new_attention_order=config.split_qkv_before_heads,
        )
        self.learn_sigma = config.learn_sigma
        scheduler_type = STRING_TO_SCHEDULER.get(config.scheduler_config.type, None)
        assert scheduler_type != None, f"Scheduler can't be {config.scheduler_config.type}, define right scheduler type in scheduler config"
        self.scheduler = scheduler_type(config.scheduler_config)
        loss_type = STRING_TO_LOSS.get(config.loss, None)
        assert loss_type != None, f"Loss can't be {config.loss}, define right loss in the config"
        self.loss = loss_type()

    def forward(self, volume, mask, return_loss=True) -> ModelOutput:
        condition = torch.mul(volume, mask)
        t, weights = self.scheduler.sample_timesteps(volume.shape[0])
        t, weights = t.to(volume.device), weights.to(volume.device)
        noise = torch.randn_like(volume).to(volume.device)
        x_t = self.scheduler.add_noise(volume, noise, t)
        model_output = self.eps_model(volume, t, condition)
        
        if self.learn_sigma:
            model_output, model_var_values = torch.split(model_output, volume.shape[1], dim=1)
        loss = self.loss(model_output, noise)
        output = ModelOutput()
        output["loss"] = loss
        return output

    def inference(self, x: Dict, print_output: bool = False) -> Dict:
        with torch.no_grad():
            mask = x["mask"]
            condition = torch.mul(x["volume"], mask)
            x = torch.randn_like(x["volume"])
            for t in tqdm(list(range(self.scheduler.num_timesteps))[::-1]):
                time = x.new_full((x.shape[0],), t, dtype=torch.long)
                xt = self.eps_model(x, time, condition)[0]
                x = self.scheduler.step(xt, t, x)
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
    
