import segmentation_models_pytorch as smp
import torch

class DeepLabV3Plus(smp.DeepLabV3Plus):
    r"""DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    """

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
