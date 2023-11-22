from torch import nn
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

    def forward(self, **kwargs):
        B, C, H, W = kwargs["pixel_values"].shape

        outputs = self.clipseg(**kwargs)
        return outputs.logits.view(B, 1, H, W)
