from typing import Callable
from monai.losses import DiceLoss
from monai.utils import LossReduction
import torch
import torchvision 

class TunedDiceLoss(DiceLoss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 0.00001,
        smooth_dr: float = 0.00001,
        batch: bool = False,
    ) -> None:
        super().__init__(
            include_background,
            to_onehot_y,
            sigmoid,
            softmax,
            other_act,
            squared_pred,
            jaccard,
            reduction,
            smooth_nr,
            smooth_dr,
            batch,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        losses = []
        for target_msk, input_msk in zip(target, input):
            usable_planes_idx = []
            for plane_idx, plane in enumerate(target_msk):
                if 1 in plane.unique():
                    # Only compute loss if there is a mask in this
                    usable_planes_idx.append(plane_idx)
            target_msk = target_msk[usable_planes_idx]
            input_msk = input_msk[usable_planes_idx]
            
            loss = super().forward(input_msk.unsqueeze(0), target_msk.unsqueeze(0))
            losses.append(loss)
        
        return sum(losses) / len(losses)
