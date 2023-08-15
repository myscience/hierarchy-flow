# Hierarchy Flow in Easy PyTorch

This repo contains the _unofficial_ implementation for [_Hierarchy Flow For High-Fidelity Image-to-Image Translation_ Fan et al. (2023)](http://arxiv.org/abs/2308.06909), which I developed as a nice entry into the world of _normalizing flows_.

The authors release their official implementation which can be found [here](https://github.com/WeichenFan/HierarchyFlow/tree/main).

# Usage

```python
import torch
from src.hflow import HierarchyFlow

flow = HierarchyFlow(
    inp_channels=3,
    flow_channel_mult=[10, 4, 4], # Set channel mult of the hierarchy convs
    feat_channel_mult=[3, 3, 3],  # Set the channel mult factors of the style convs 
    pad_size = 10, # Input pad size
    pad_mode = 'reflect',
    style_out_dim = 8, # Number of channel of final style features
    style_conv_kw = [
        {'kernel_size' : 7, 'stride' : 1, 'padding' : 3},
        *[{'kernel_size' : 4, 'stride' : 2, 'padding' : 1}] * 2
    ] # Parameter for style convolutional layers, should match length of feat_channel_mult
)

x = torch.randn(1, 3, 256, 256)
y = flow(x)
```