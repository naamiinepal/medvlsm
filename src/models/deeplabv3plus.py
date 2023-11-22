from typing import Optional
import segmentation_models_pytorch as smp
import torch

class DeepLabV3Plus(smp.DeepLabV3Plus):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: str | None = None,
        upsampling: int = 4,
        aux_params: dict | None = None,
    ):
        super().__init__(
            encoder_name,
            encoder_depth,
            encoder_weights,
            encoder_output_stride,
            decoder_channels,
            decoder_atrous_rates,
            in_channels,
            classes,
            activation,
            upsampling,
            aux_params,
        )

    def forward(self, **kwargs) -> torch.Tensor:
        pixel_values = kwargs["pixel_values"]
        return super().forward(pixel_values)
