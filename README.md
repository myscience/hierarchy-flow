# Hierarchy Flow in Easy PyTorch

This repo contains the _unofficial_ implementation for [_Hierarchy Flow For High-Fidelity Image-to-Image Translation_ Fan et al. (2023)](http://arxiv.org/abs/2308.06909), which I developed as a nice entry into the world of _normalizing flows_.

The authors release their official implementation which can be found [here](https://github.com/WeichenFan/HierarchyFlow/tree/main).

# Usage

```python
import torch
from src.hflow import HierarchicalFlow

flow = HierarchicalFlow(
    inp_channels=3,
    out_channels=[30, 120],
    pad_size=10,
)

x = torch.randn(1, 3, 256, 256)
y = flow(x)
```