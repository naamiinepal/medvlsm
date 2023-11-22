from typing import Optional
from torch import Tensor
from torch.nn import BCELoss as _BCELoss


class BCELoss(_BCELoss):
    def __init__(self, weight: Tensor | None = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target.float())