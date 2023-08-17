import torch
import unittest

import torch.nn as nn
from torch import Tensor

from src.losses import StyleLoss

class StyleLossTest(unittest.TestCase):
    def setUp(self) -> None:

        self.criterion = StyleLoss(
            enc_depth=(5, 12, 19, 32),
            backbone='vgg19_bn',
        )

        # Create a dummy input of correct shape
        self.input_shape = (2, 3, 256, 256)

        self.orig_img = torch.randn(*self.input_shape, requires_grad=True)
        self.targ_sty = torch.randn(*self.input_shape, requires_grad=True)
        self.pred_img = torch.randn(*self.input_shape, requires_grad=True)

    def test_default_is_relu(self):
        layers = self.criterion.backbone.layers
    
        self.assertTrue(all([isinstance(l, nn.ReLU) for l in layers]))

    def test_forward(self):
        loss = self.criterion(self.orig_img, self.targ_sty, self.pred_img)

        # Check the output shape
        self.assertTrue(loss > 0)

    def test_backward(self):
        loss : Tensor = self.criterion(self.orig_img, self.targ_sty, self.pred_img)

        loss.backward()

        # Check the output shape
        self.assertTrue(loss > 0)
    
