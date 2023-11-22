from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .clip import build_model
from .layers import FPN, Projector, TransformerDecoder, _int_3_tup


class CRIS(nn.Module):
    def __init__(
        self,
        clip_pretrain: str,
        word_len: int,
        fpn_in: _int_3_tup,
        fpn_out: _int_3_tup,
        vis_dim: int,
        word_dim: int,
        num_layers: int,
        num_head: int,
        dim_ffn: int,
        dropout: float,
        intermediate: bool,
        img_size: int = 416,
        freeze_encoder: bool = True,
        cris_pretrain: Optional[str] = None,
    ):
        super().__init__()

        self.img_size = img_size

        # Vision & Text Encoder
        clip_model = torch.jit.load(clip_pretrain, map_location="cpu")
        self.backbone = build_model(clip_model.state_dict(), word_len).float()

        self.backbone.requires_grad_(not freeze_encoder)

        # Multi-Modal FPN
        self.neck = FPN(in_channels=fpn_in, out_channels=fpn_out)

        # Decoder
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            d_model=vis_dim,
            nhead=num_head,
            dim_ffn=dim_ffn,
            dropout=dropout,
            return_intermediate=intermediate,
        )

        # Projector
        self.proj = Projector(word_dim, vis_dim // 2, 3)

        if cris_pretrain is not None:
            self.load_state_dict(
                torch.load(cris_pretrain, map_location="cpu"), strict=True
            )

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        """
        img: b, 3, h, w
        word: b, words
        word_mask: b, words
        mask: b, 1, h, w
        """
        # padding mask used in decoder
        pad_mask = torch.zeros_like(input_ids).masked_fill_(input_ids == 0, 1).bool()

        # vis: C3 / C4 / C5
        # input_ids: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(pixel_values)
        input_ids, state = self.backbone.encode_text(input_ids)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, input_ids, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        pred = F.interpolate(pred, self.img_size, mode="bicubic", align_corners=True)

        return pred
