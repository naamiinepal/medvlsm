from typing import List, Optional, Union, Any
import segmentation_models_pytorch as smp


class UNetPP(smp.UnetPlusPlus):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: str | None = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: str | Any | None = None,
        aux_params: dict | None = None,
    ):
        super().__init__(
            encoder_name,
            encoder_depth,
            encoder_weights,
            decoder_use_batchnorm,
            decoder_channels,
            decoder_attention_type,
            in_channels,
            classes,
            activation,
            aux_params,
        )
    

    def forward(self, **kwargs):
        pixel_values = kwargs["pixel_values"]
        return super().forward(pixel_values)

