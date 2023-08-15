
import torch.nn as nn
from torch import Tensor

from typing import List

from .utils import default
from .utils import flatten

class FeatureRecorder:
    ''' 
        Basic feature recording class that implements a PyTorch hook to
        acquire a layer activations as they get processed by the network
    '''
    
    def __init__(
        self,
        names : List[str],
    ) -> None:
        
        self.names = names
        self.feats = {l : [] for l in names}
        
    def __call__(self, module : nn.Module, inp : Tensor, out : Tensor) -> None:
        # Detach layer output from PyTorch graph
        data = out.detach()

        # Get the module name
        layer = module.name

        self.feats[layer].append(data)

    def clean(
        self,
        names : List[str] | None = None,
    ) -> None:
        self.names = default(names, self.names)
        self.feats = {k : [] for k in self.names}

class Traced(nn.Module):
    '''
        A wrapper class of a Torch Module whose intermediate activations
        are traced (recorded) via forward hooks at chosen depth.
    '''

    def __init__(
        self,
        module : nn.Module,
        depths : List[int],
    ) -> None:
        super().__init__()

        self.module = module
        self.depths = depths

        # Get the list of layers to be traced
        self.layers = [l for depth, l in enumerate(flatten(module)) if depth in depths]

        for layer, name in zip(self.layers, self.names): layer.name = name

        # Initialize the feature recorder
        self.names = [f'enc_{d}' for d in depths]
        self.recorder = FeatureRecorder(self.names)

        # Register the forward hooks for each layered targeted as 'traced'
        self.hook_handles = [l.register_forward_hook(self.recorder) for l in self.layers]
    
    @property
    def features(self) -> List[Tensor]:
        return [self.recorder.feats[k] for k in self.names]

    def forward(self, inp : Tensor, auto_clean : bool = True) -> List[Tensor]:
        # Propagate the input into the network
        _ = self.module(inp)

        # Collect the features
        feats = self.features

        if auto_clean: self.clean()

        return feats

    def clean(self) -> None:
        self.recorder.clean()

        for h in self.hook_handles: h.remove()

    def __str__(self) -> str:
        msg = 'Tracing module: \n'
        msg += f'{self.module} \n'
        msg += f'Traced layers: {self.layers} \n'
        msg += f'Traced depths: {self.depths} \n'
        return msg