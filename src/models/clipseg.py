from typing import Optional

from torch import nn
import torch
from transformers import CLIPSegForImageSegmentation


class CLIPSeg(nn.Module):
    def __init__(
        self,
        clipseg_hf_api: str,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__()

        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_hf_api)

        self.clipseg.clip.requires_grad_(not freeze_encoder)
        self.clipseg.decoder.requires_grad_(not freeze_decoder)

    def forward(
        self, 
        input_ids:torch.Tensor, 
        pixel_values:torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        **kwargs
    ):

        B, C, H, W = pixel_values.shape
        outputs = self.clipseg(input_ids, pixel_values, attention_mask, **kwargs)

        return outputs.logits.view(B, 1, H, W)
