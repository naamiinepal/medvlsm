from typing import Optional

from torch import nn
import torch
from transformers import CLIPSegForImageSegmentation


class CLIPSeg(nn.Module):
    r"""CLIPSeg Official implementation from HuggingFace.

    Args:
        clipseg_hf_api (str): HuggingFace api to import the CLIPSeg implementation; Eg:'CIDAS/clipseg-rd64-refined'
        freeze_encoder (bool): Whether or not to freeze the encoders of pretrained CLIPSeg; Default is False.
        freeze_decoder (bool): Whether or not to freeze the decoder of pretrained CLIPSeg; Default is False.
    """
    def __init__(
        self,
        clipseg_hf_api: str,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
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
    ) -> torch.Tensor:
        r"""
        Args:
            pixel_values: Normalized image tensor.
            input_ids: Tokenized text input.
            attention_mask: Mask for token inputs, used in the attention layers.

        Returns: Tensor with segmentation logits
        """
        
        B, C, H, W = pixel_values.shape
        outputs = self.clipseg(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask
        )

        return outputs.logits.view(B, 1, H, W)
