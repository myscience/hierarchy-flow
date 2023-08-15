import torch
import torch.nn as nn

from torch import Tensor
from typing import Any, Tuple, List

from einops import rearrange

def exists(var : Any | None) -> bool:
    return var is not None

def default(var : Any | None, val : Any) -> Any:
    return val if var is None else var

def flatten(model : nn.Module, exclude : List[nn.Module] = []) -> List[nn.Module]:
    flattened = [flatten(children) for children in model.children()]
    res = [model] if list(model.children()) == [] else []

    for c in flattened: res += c
    
    return res


def compute_stat(feat : Tensor, eps : float = 1e-5) -> Tuple[Tensor, Tensor]:
    # Check input dimension
    bs, c, h, w = feat.shape

    var = rearrange(feat, 'b c h w -> b c (h w)').var(dim=2) + eps
    std = rearrange(var.sqrt(), 'b c -> b c 1 1')

    mean = rearrange(feat, 'b c h w -> b c (h w)').mean(dim=2)
    mean = rearrange(mean, 'b c -> b c 1 1')

    return mean, std

class ReversibleConcat(nn.Module):
    '''
    '''

    def __init__(self, cat_dim : int = 0) -> None:
        super().__init__()

        self.dim = cat_dim

    def forward(self, inp : Tensor) -> Tensor:
        return torch.cat(inp, dim=self.dim)
    
    def reverse(self, inp : Tensor) -> Tensor:
        return inp[-1]
    

class ReversiblePad2d(nn.Module):
    '''
        This is a reversible Pad layer.
    '''

    def __init__(self, pad_size : int = 10, mode = 'reflect') -> None:
        super().__init__()

        self.pad_size = pad_size

        match mode:
            case 'reflect':
                self.padding = nn.ReflectionPad2d(pad_size)
            case _:
                raise ValueError(f'Unknown pad mode {mode}.')

    def forward(self, x : Tensor) -> Tensor:
        return self.padding(x)

    def reverse(self, x : Tensor, *args) -> Tensor:
        *_, w, h, p = *x.shape, self.pad_size

        return x[..., p:h-p, p:w-p]
    
class AdaIN(nn.Module):
    '''
    '''

    def __init__(self) -> None:
        super().__init__()

    def forward(self, subj : Tensor, feat_mean : Tensor, feat_std : Tensor) -> Tensor:
        # Get subject mean and standard deviation
        subj_mean, subj_std = compute_stat(subj)

        norm_feat = (subj - subj_mean) / subj_std

        return norm_feat * feat_std + feat_mean