from typing import List

import torch
from torch import nn
from transformers import CLIPModel
from .side_adapter import RegionWiseSideAdapterNetwork

class SAN(nn.Module):

    def __init__(
        self,
        clip_visual_extractor: nn.Module,
        clip_rec_head: nn.Module,
        side_adapter_network: RegionWiseSideAdapterNetwork,
        size_divisibility: int,
        clip_resolution: float = 0.5,
        sem_seg_postprocess_before_inference: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.clip_resolution = clip_resolution
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

        self.side_adapter_network = side_adapter_network
        self.clip_visual_extractor = clip_visual_extractor
        self.clip_rec_head = clip_rec_head

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask:torch.Tensor, **kwargs):
        
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True)

        vision_hidden_states = vision_outputs.hidden_states

        mask_preds, attn_biases = self.side_adapter_network(pixel_values, vision_hidden_states)

        text_features = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        # !! Could be optimized to run in parallel.
        mask_embs = [
            self.clip_rec_head(vision_hidden_states, attn_bias, normalize=True)
            for attn_bias in attn_biases
        ]  # [B,N,C]
        mask_logits = [
            torch.einsum("bqc,nc->bqn", mask_emb, text_features)
            for mask_emb in mask_embs
        ]

        return {"pred_logits": mask_logits[-1], "pred_masks": mask_preds[-1]}




