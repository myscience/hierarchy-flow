import torch
import torch.nn as nn
from torchvision.models import get_model
from torch.nn.functional import mse_loss

from typing import List
from torch import Tensor

from .recorder import Traced
from .utils import compute_stat

def style_loss(targ : Tensor, pred : Tensor, k : float = 0.) -> Tensor:
    bs, c, *_ = targ.shape

    # Compare mean & std of target and pred tensor
    targ_mean, targ_std = compute_stat(targ)
    pred_mean, pred_std = compute_stat(pred)

    loss_mean = mse_loss(targ_mean, pred_mean, reduction='none')
    loss_std  = mse_loss(targ_std , pred_std , reduction='none')

    if k > 0:
        loss_mean, sort_idx = torch.sort(loss_mean, dim=1)

        loss_mean[:, int(c * k):] = 0
        loss_std[:, sort_idx[:, int(c * k):]] = 0

    return (loss_mean + loss_std).mean()

def content_loss(targ : Tensor, pred : Tensor) -> Tensor:
    # Compare mean & std of target and pred tensor
    targ_mean, targ_std = compute_stat(targ)
    pred_mean, pred_std = compute_stat(pred)

    norm_targ = (targ - targ_mean) / targ_std
    norm_pred = (pred - pred_mean) / pred_std

    return mse_loss(norm_targ, norm_pred)

class StyleLoss(nn.Module):
    '''
        This module implements the style loss used for model
        training in the paper:
        `Hierarchy Flow For High-Fidelity Image-to-Image Translation`
        Fan et al. (2023) (arxiv:2308.06909).

        This loss is a combination of a standard VGG-19 (style) loss
        and and an original (modification of) content loss. The tradeoff
        between content-preserving and style-preserving is controlled by
        the `content_weight` parameter.

        Args:
        - enc_depth (List[int]): List of integers representing the
          backbone layer depth to use as feature encoders.
        - backbone (str): Torchvision model to use as feature extractor
        - content_weight (float): The weight of the content loss.
    '''

    def __init__(
        self,
        enc_depth : List[int] = (5, 12, 19, 32),
        backbone : str = 'vgg19_bn',
        content_weight : float = .8,
    ) -> None:
        super().__init__()

        backbone = get_model(backbone, weights='DEFAULT')
        backbone = Traced(backbone, enc_depth)

        self.backbone = backbone

        self.content_weight = content_weight

    def forward(
        self,
        orig_img : Tensor,
        targ_sty : Tensor,
        pred_img : Tensor,
        align_k : float = 0.8    
    ) -> Tensor:
        '''
        '''
        # Get features from the backbone model for the
        # original content image
        (*_, orig_feat) = self.backbone(orig_img)

        # Get the style features for both the reference
        # and produced images
        targ_feats = self.backbone(targ_sty)
        pred_feats = self.backbone(pred_img)

        # Compute the style & content loss
        loss_style = sum([style_loss(f1, f2, k=align_k) for f1, f2 in zip(targ_feats, pred_feats)])
        loss_content = content_loss(orig_feat, pred_feats[-1])

        # Combine the style and content loss
        loss = loss_content + self.content_weight * loss_style

        return loss


