from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import VisionTransformer
from transformers.models.clip.modeling_clip import CLIPVisionTransformer 

def downsample2d(src, target_shape, method="nearest"):
    # src: [N,C,H,W]
    # target_shape: [H',W']
    # return: [N,C,H',W']
    if method in ["bicubic", "bilinear", "nearest"]:
        src = F.interpolate(src, size=target_shape, mode=method, align_corners=False)
    elif method == "avg":
        src = F.adaptive_avg_pool2d(src, output_size=target_shape)
    elif method == "max":
        src = F.adaptive_max_pool2d(src, output_size=target_shape)
    return src

def cross_attn_layer(self: ResidualAttentionBlock, x, mem, attn_bias):
    # x: [K,N,C]
    # mem: [L,N,C]
    # attn_bias: [N*num_head,K,L]
    # return: [K,N,C]
    q_x = self.ln_1(x)
    k_x = v_x = self.ln_1(mem)
    x = x + self.ls_1(
        cross_attn_with_self_bias(self.attn, q_x, k_x, v_x, attn_mask=attn_bias)[0]
    )
    x = x + self.ls_2(self.mlp(self.ln_2(x)))
    return x

class RecWithAttnbiasHead(nn.Module):
    def __init__(
        self,
        visual_encoder: CLIPVisionTransformer,
        visual_proj: nn.Linear,
        first_layer_idx: int = 0,
        frozen_exclude: List[str] = [],
        sos_token_format: str = "cls_token",
        sos_token_num: int = 1,
        cross_attn: bool = True,
        downsample_method: str = "bilinear",
    ):
        super().__init__()
        self.output_dim = visual_encoder.config.projection_dim
        self.first_layer_idx = first_layer_idx
        self.cross_attn = cross_attn
        self.downsample_method = downsample_method

        if first_layer_idx < 0:
            raise NotImplementedError("first_layer_idx < 0 is not implemented yet.")
        self.resblocks = visual_encoder.encoder.layers[first_layer_idx:]
        self.ln_post = visual_encoder.post_layernorm
        self.proj = visual_proj

        self.sos_token_format = sos_token_format
        self.sos_token_num = sos_token_num
        self.frozen_exclude = frozen_exclude

        if sos_token_format in ["learnable_token", "pos_embedding"]:
            self.sos_token = nn.Parameter(
                torch.randn(sos_token_num, 1, self.proj.shape[0])
            )
            nn.init.normal_(self.sos_token, std=0.02)
            self.frozen_exclude.append("sos_token")
        self._freeze(self.frozen_exclude)

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = False

    def forward(self, features, attn_bias, normalize: bool = False):
        # construct clip shadow features.
        cls_token = features[f"{self.first_layer_idx}_cls_token"]  # 1,n,c
        pix_feat = features[self.first_layer_idx]  # n,c,h,w
        n, c, h, w = pix_feat.shape
        x = torch.cat(
            [cls_token, pix_feat.reshape(n, c, -1).permute(2, 0, 1)]
        )  # 1+l,n,c

        # construct sos token.
        if self.sos_token_format == "cls_token":
            sos_token = cls_token.repeat(self.sos_token_num, 1, 1)
        elif self.sos_token_format == "learnable_token":
            sos_token = self.sos_token.expand(-1, n, -1)
        elif self.sos_token_format == "pos_embedding":
            sos_token = self.sos_token.expand(-1, n, -1) + cls_token

        # construct attn biases.
        attn_biases = self._build_attn_biases(attn_bias, target_shape=(h, w))
        if self.cross_attn:
            for i, resblock in enumerate(self.resblocks):
                if self.cross_attn:
                    sos_token = cross_attn_layer(
                        resblock,
                        sos_token,
                        x[1:,],
                        attn_biases[i],
                    )
                    if i < len(self.resblocks) - 1:
                        x = resblock(x)
        else:
            x = torch.cat([sos_token, x], dim=0)
            for i, resblock in enumerate(self.resblocks):
                x = resblock(x, attn_mask=attn_biases[i])
            sos_token = x[: self.sos_token_num]

        sos_token = sos_token.permute(1, 0, 2)  # LND -> NLD

        sos_token = self.ln_post(sos_token)

        if self.proj is not None:
            sos_token = sos_token @ self.proj
        if normalize:
            sos_token = F.normalize(sos_token, dim=-1)
        return sos_token

    def _build_attn_biases(self, attn_biases, target_shape):
        formatted_attn_biases = []
        for attn_bias in attn_biases:
            # convert it to proper format: N*num_head,L,L
            # attn_bias: [N, num_head/1, num_sos,H,W]
            n, num_head, num_sos, h, w = attn_bias.shape
            # reshape and downsample
            attn_bias = downsample2d(
                attn_bias.reshape(n, num_head * num_sos, h, w),
                target_shape,
                method=self.downsample_method,
            )
            attn_bias = attn_bias.reshape(n, num_head, num_sos, *target_shape)
            true_num_head = self.resblocks[0].attn.num_heads
            assert (
                num_head == 1 or num_head == true_num_head
            ), f"num_head={num_head} is not supported."
            if num_head == 1:
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            attn_bias = attn_bias.reshape(n * true_num_head, num_sos, -1)
            L = attn_bias.shape[-1]
            if self.cross_attn:
                # [n*num_head, num_sos, L]
                formatted_attn_biases.append(attn_bias)
            else:
                # [n*num_head, num_sos+1+L, num_sos+1+L]
                new_attn_bias = attn_bias.new_zeros(num_sos + 1 + L, num_sos + 1 + L)
                new_attn_bias[:, :num_sos] = -100
                new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0
                new_attn_bias[:num_sos, num_sos] = -100
                new_attn_bias = (
                    new_attn_bias[None, ...].expand(n * true_num_head, -1, -1).clone()
                )
                new_attn_bias[..., :num_sos, -L:] = attn_bias
                formatted_attn_biases.append(new_attn_bias)

        if len(formatted_attn_biases) == 1:
            formatted_attn_biases = [formatted_attn_biases[0] for _ in self.resblocks]
        return formatted_attn_biases