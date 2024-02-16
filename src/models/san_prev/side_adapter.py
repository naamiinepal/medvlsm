from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import VisionTransformer

class RegionWiseSideAdapterNetwork(nn.Module):

    def __init__(
        self,
        vit_model: VisionTransformer,
        fusion_layers: nn.ModuleList,
        mask_decoder: nn.Module,
        num_queries: int,
        fusion_map: Dict[int, int],
        deep_supervision_idxs: List[int],
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if vit_model.cls_token is not None:
            vit_model.pos_embed = nn.Parameter(vit_model.pos_embed[:, 1:, ...])
        del vit_model.cls_token
        vit_model.cls_token = None
        # delete out norm
        del vit_model.norm
        vit_model.norm = nn.Identity()
        self.vit_model = vit_model

        self.num_queries = num_queries
        self.num_features = vit_model.num_features
        # add query token
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, self.num_features))
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, num_queries, self.num_features)
        )
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
        self.fusion_layers = fusion_layers
        self.fusion_map = fusion_map
        self.mask_decoder = mask_decoder
        # for training
        self.deep_supervision_idxs = deep_supervision_idxs

    
    def forward(self, image: torch.Tensor, clip_features: List[torch.Tensor]):
        x, (h, w) = self.vit_model.patch_embed(image)
        L = x.shape[1]  # token length
        pos_embed = self.vit_model.pos_embed
        ori_h, ori_w = self.vit_model.patch_embed.grid_size
        if pos_embed.shape[1] != L:
            pos_embed = (
                F.interpolate(
                    pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
                    size=[h, w],
                    mode="bicubic",
                    align_corners=False,
                )
                .flatten(2)
                .permute(0, 2, 1)
            )
        pos_embed = torch.cat(
            [self.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed], dim=1
        )
        x = torch.cat(
            [self.query_embed.expand(x.shape[0], -1, -1), x],
            dim=1,
        )  # B, Q+L, C
        x = x + pos_embed
        x = self.vit_model.norm_pre(x)
        x = self.fuse(0, x, clip_features, (h, w))
        outs = []
        for i, blk in enumerate(self.vit_model.blocks, start=1):
            x = blk(x)
            x = self.fuse(i, x, clip_features, (h, w))
            if i in self.deep_supervision_idxs:
                outs.append(
                    {
                        "query": x[:, :-L, ...],
                        "x": x[:, -L:, ...]
                        .permute(0, 2, 1)
                        .reshape(x.shape[0], x.shape[-1], h, w),
                    }
                )

            if i < len(self.vit_model.blocks):
                x = x + pos_embed

        return outs

    def fuse(
        self,
        block_idx: int,
        x: torch.Tensor,
        clip_features: List[torch.Tensor],
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if block_idx in self.fusion_map:
            src_idx = self.fusion_map[block_idx]
            L = spatial_shape[0] * spatial_shape[1]
            x = torch.cat(
                [
                    x[:, :-L, ...],
                    self.fusion_layers[f"layer_{block_idx}"](
                        x[:, -L:, ...], clip_features[src_idx], spatial_shape
                    ),
                ],
                dim=1,
            )
            # log_first_n(
            #     logging.INFO,
            #     f"fuse clip {src_idx} to {block_idx}",
            #     len(self.fusion_map),
            # )
        return x