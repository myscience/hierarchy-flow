import torch
import unittest

from src.losses import StyleLoss

class StyleLossTest(unittest.TestCase):
    def setUp(self) -> None:

        self.criterion = StyleLoss(
            enc_depth=(3, 10, 17, 30),
            backbone='vgg19_bn',
        )

        # Create a dummy input of correct shape
        self.input_shape = (2, 3, 256, 256)

        self.orig_img = torch.randn(*self.input_shape)
        self.targ_sty = torch.randn(*self.input_shape)
        self.pred_img = torch.randn(*self.input_shape)
    
    def test_forward(self):
        loss = self.criterion(self.orig_img, self.targ_sty, self.pred_img)

        # Check the output shape
        self.assertTrue(loss > 0)
