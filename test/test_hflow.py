import torch
import unittest

from torch import Tensor

from src.hflow import HierarchyFlow
from src.losses import StyleLoss

class HFlowTest(unittest.TestCase):
    def setUp(self) -> None:

        self.flow = HierarchyFlow(
            inp_channels=3,
            flow_channel_mult=[10, 4, 4], 
            feat_channel_mult=[3, 1, 1],  
            pad_size = 10, 
            pad_mode = 'reflect',
            style_out_dim = 8,
            style_conv_kw = [
                {'kernel_size' : 7, 'stride' : 1, 'padding' : 3},
                *[{'kernel_size' : 4, 'stride' : 2, 'padding' : 1}] * 2
            ]
        )

        self.criterion = StyleLoss(
            enc_depth=(5, 12, 19, 32),
            backbone='vgg19_bn',
        )

        # Create a dummy input of correct shape
        self.input_shape = (2, 3, 256, 256)

        self.dummy_subj  = torch.randn(*self.input_shape)
        self.dummy_style = torch.randn(*self.input_shape)
    
    def test_forward(self):
        output = self.flow(self.dummy_subj, self.dummy_style)
        
        # Check the output shape
        self.assertEqual(output.shape, self.input_shape)

    def test_loss_backward(self):
        pred_img = self.flow(self.dummy_subj, self.dummy_style)
        
        loss : Tensor = self.criterion(self.dummy_subj, self.dummy_style, pred_img)

        loss.backward()

        # Check the output shape
        self.assertTrue(loss > 0)

