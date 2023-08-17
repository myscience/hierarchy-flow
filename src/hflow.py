import torch
import torch.nn as nn

from .utils import default
from .utils import ReversiblePad2d
from .blocks import HierarchyBlock

from typing import List
from torch import Tensor
from itertools import pairwise, repeat

class HierarchyFlow(nn.Module):
    '''
        Unofficial PyTorch implementation of:
        `Hierarchy Flow For High-Fidelity Image-to-Image Translation`
        Fan et al. (2023) (arxiv:2308.06909).
    '''
    def __init__(
        self,
        inp_channels : int = 3,
        flow_channel_mult : List[int] = [10, 3],
        feat_channel_mult : List[int] = [3, 3],
        pad_size : int = 10,
        pad_mode : str = 'reflect',
        style_out_dim : int = 8,
        style_conv_kw : dict | None = None,
    ):
        super(HierarchyFlow, self).__init__()

        style_conv_kw = default(style_conv_kw, repeat({'kernel_size' : 3}))

        self.inp_channels = inp_channels
        self.flow_channel_mult = torch.cumprod(torch.tensor(flow_channel_mult), dim=0)
        self.feat_channel_mult = torch.cumprod(torch.tensor(feat_channel_mult), dim=0)
        self.pad_size = pad_size

        flow_channels = (inp_channels, *[inp_channels * mult for mult in self.flow_channel_mult])
        feat_channels = (inp_channels, *[inp_channels * mult for mult in self.feat_channel_mult])

        self.layers = nn.ModuleList([
                ReversiblePad2d(pad_size, pad_mode=pad_mode),
                *[HierarchyBlock(
                    inp_chn, out_chn,
                    mlp_inp_dim=style_out_dim)
                for inp_chn, out_chn in pairwise(flow_channels)]
            ]
        )

        self.style_block = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(inp_chn, out_chn, **style),
                nn.ReLU(),
            ) for (inp_chn, out_chn), style in zip(pairwise(feat_channels), style_conv_kw)],
            nn.AdaptiveAvgPool2d(1), # global average pooling
            nn.Conv2d(feat_channels[-1], style_out_dim, 1, 1, 0)
        )

    def forward(
        self,
        img_subj  : Tensor,
        img_style : Tensor,
    ) -> Tensor:
        '''
        
        '''
        # Compute the abstract features from reference style
        feat = self.style_block(img_style)

        # This is the forward pass, where we extract subject
        # style features, before image translations
        subj = img_subj
        for block in self.layers:
            subj = block(subj)

        # This is the reverse pass, where we use the provided
        # style to translate the image content
        for block in self.layers[::-1]:
            subj = block.reverse(subj, feat)

        return subj
