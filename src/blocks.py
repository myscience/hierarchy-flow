import torch
import torch.nn as nn

from .utils import AdaIN
from .utils import default
from .utils import ReversibleConcat

from torch import Tensor
from typing import List
from einops import rearrange
from itertools import pairwise

class StyleMLP(nn.Module):
    '''
        Basic Multi-Layer Perceptron.
    '''

    def __init__(
        self,
        inp_dim : int,
        hid_dim : int | List[int] | None = None,
        out_dim : int | None = None,
        act_fn : str = 'relu',
    ) -> None:
        super().__init__()

        hid_dim = default(hid_dim, [])
        out_dim = default(out_dim, inp_dim)

        if isinstance(hid_dim, int): hid_dim = [hid_dim]

        match act_fn:
            case 'relu': ActFn = nn.ReLU
            case 'selu': ActFn = nn.SELU
            case 'tanh': ActFn = nn.Tanh
            case 'none': ActFn = nn.Identity
            case _: raise ValueError(f'Unknown activation function: {act_fn}')

        layers = []
        for dim_in, dim_out in pairwise([inp_dim, *hid_dim, out_dim]):
            layers.extend([
                nn.Linear(dim_in, dim_out),
                ActFn(),
            ])

        self.layers = nn.Sequential(
            *layers
        )

    def forward(self, inp : Tensor) -> Tensor:
        inp = rearrange(inp, 'b ... -> b (...)')
        out = self.layers(inp)

        return rearrange(out, 'b c -> b c 1 1')

class HierarchyBlock(nn.Module):
    '''
    '''

    def __init__(
        self,
        inp_channel : int,
        out_channel : int,
        mlp_inp_dim : int,
        mlp_n_layer : int = 3,
        mlp_actv_fn : str = 'relu',
        channel_mult : int = 2,
        reverse_fact : float = .5,
    ) -> None:
        super().__init__()

        self.inp_channel = inp_channel
        self.out_channel = out_channel

        hid_channel = inp_channel * channel_mult
        self.alpha = nn.Parameter(torch.tensor(reverse_fact), requires_grad=True)

        # Network used in forward pass
        self.affine = nn.Sequential(
            nn.Conv2d(inp_channel, hid_channel, 3, 1, 1),
            nn.InstanceNorm2d(hid_channel),
            nn.ReLU(),
            nn.Conv2d(hid_channel, hid_channel, 3, 1, 1),
            nn.InstanceNorm2d(hid_channel),
            nn.ReLU(),
            nn.Conv2d(hid_channel, out_channel, 3, 1, 1),
            nn.ReLU(),
        )

        self.proj_out = ReversibleConcat(dim=1)

        # Networks used in reverse pass to in-paint style
        self.ada = AdaIN()
        self.mlp = StyleMLP(
            inp_dim=mlp_inp_dim,
            hid_dim=[out_channel * 3] * mlp_n_layer,
            # The x2 in out_dim is because we predict
            # both mean and std for each channel
            out_dim=out_channel * 2,
            act_fn=mlp_actv_fn,
        )

    def forward(self, img : Tensor) -> Tensor:
        '''
        '''
        bs, c, h, w = img.shape

        self.feat : Tensor = self.affine(img)

        # Divide the feature into chunks of size [channel].
        feats = rearrange(self.feat, 'b (n c) h w -> n b c h w', c = c)
        feats = torch.stack([img, *(-feats)], dim=0)

        _, *feats = torch.cumsum(feats, dim=0)

        out = self.proj_out(feats)

        return out

    def reverse(self, subj : Tensor, feat : Tensor) -> Tensor:
        '''
        '''

        c = self.inp_channel

        # Get the predicted mean and std from reference style
        mean, std = self.mlp(feat).chunk(2, dim=1)

        # Use AdaIN to in-paint target style
        subj = self.ada(subj, mean, std)

        feat, *feats = rearrange(self.feat, 'b (n c) h w -> n b c h w', c = c).flipud()
        inp,  *inps  = rearrange(subj,      'b (n c) h w -> n b c h w', c = c).flipud()

        outs = [inp + feat]
        for feat, inp in zip(feats, inps):
            # ! NOTE: This code seems inconsistent with Algorithm (2) in the paper
            out = self.alpha * (outs[-1] + inp) + (1 - self.alpha) * feat 
            outs.append(out)

        out = self.proj_out.reverse(outs)

        return out