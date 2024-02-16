from typing import List, Dict, Any


import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.trainer.states import RunningStage
import open_clip

from .clip_utils import (
    FeatureExtractor,
    RecWithAttnbiasHead,
)
from .side_adapter import RegionwiseSideAdapterNetwork


class SAN(nn.Module):
    def __init__(
        self,
        clip_visual_extractor_cfg: Dict[str, Any],
        clip_rec_head_cfg: Dict[str, Any],
        side_adapter_network: RegionwiseSideAdapterNetwork,
        size_divisibility: int,
        asymetric_input: bool = True,
        clip_resolution: float = 0.5,
        sem_seg_postprocess_before_inference: bool = False,
    ):
        super().__init__()
        self.asymetric_input = asymetric_input
        self.clip_resolution = clip_resolution
        
        self.clip_model, _ = open_clip.create_model_from_pretrained(
            model_name="ViT-B/16", pretrained="openai"
        )
        
        self.clip_model.requires_grad_(False)
        
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility

        self.side_adapter_network = side_adapter_network
        self.clip_visual_extractor = FeatureExtractor(
            visual_encoder=self.clip_model.visual, **clip_visual_extractor_cfg
        )
        self.clip_rec_head = RecWithAttnbiasHead(
            visual_encoder=self.clip_model.visual, **clip_rec_head_cfg
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        stage: RunningStage,
        **kwargs
    ):

        if self.asymetric_input:
            clip_input = F.interpolate(
                pixel_values, scale_factor=self.clip_resolution, mode="bilinear"
            )

        clip_image_features = self.clip_visual_extractor(clip_input)
        mask_preds, attn_biases = self.side_adapter_network(
            pixel_values, clip_image_features
        )

        text_features = self.clip_model.encode_text(input_ids)
        # !! Could be optimized to run in parallel.
        mask_embs = [
            self.clip_rec_head(clip_image_features, attn_bias, normalize=True)
            for attn_bias in attn_biases
        ]  # [B,N,C]
        print("mask_embs[0].shape", mask_embs[0].shape, "text_features", text_features.shape)
        mask_logits = [
            torch.einsum("iqd,td->iqt", mask_emb, text_features)
            for mask_emb in mask_embs
        ]

        mask_logits = mask_logits[-1]
        mask_preds = mask_preds[-1]
        
        mask_preds = F.interpolate(
            mask_preds,
            size=(pixel_values.shape[-2], pixel_values.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        
        print("mask_cls.shape", mask_logits.shape, "mask_pred.shape", mask_preds.shape)
        pred_masks = torch.einsum("iqt,iqhw->thw", mask_logits, mask_preds)[:,None]
        print(pred_masks.requires_grad)
        return pred_masks
        # if stage == RunningStage.TRAINING:
        #     return {
        #         "pred_logits": mask_logits[-1],
        #         "pred_masks": mask_preds[-1],
        #     }
        # else:
        #     mask_preds = mask_preds[-1]
        #     mask_logits = mask_logits[-1]
        #     # torch.cuda.empty_cache()
        #     # Inference

        #     self.semantic_inference(mask_logits, mask_preds)

    # def semantic_inference(self, mask_cls: torch.Tensor, mask_pred: torch.Tensor):
    #     print("mask_cls.shape", mask_cls.shape, "mask_pred.shape", mask_pred.shape)
    #     mask_cls = mask_cls.sigmoid()
    #     mask_pred = mask_pred.sigmoid()
    #     print("mask_cls.shape", mask_cls.shape, "mask_pred.shape", mask_pred.shape)
    #     semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
    #     return semseg
