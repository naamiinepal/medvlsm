import math
from typing import List, Optional

import open_clip
import torch
from open_clip.hf_model import ClsPooler
from torch import nn
from transformers.models.clipseg.modeling_clipseg import CLIPSegDecoder
from transformers import CLIPSegConfig
from torchvision.transforms.functional import resize


class BiomedCLIPSeg(nn.Module):
    def __init__(
        self,
        extract_layers: List[int] = [3, 6, 9],
        cond_layer: int = 0,
        biomedclip_hf_api: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        reduce_dim: int = 64,
        n_heads: int = 4,
        extra_blocks: int = 0,
        image_size: int = 224,
        mask_size: int = 224,
        text_supervision: bool = True,
    ) -> None:
        super(BiomedCLIPSeg, self).__init__()

        self.image_size = image_size
        self.mask_size = mask_size
        self.text_supervision = text_supervision

        self.biomedclip_model, _ = open_clip.create_model_from_pretrained(
            biomedclip_hf_api
        )
        # self.biomedclip_model.
        for p in self.biomedclip_model.parameters():
            p.requires_grad = False

        self.clipseg_config = CLIPSegConfig()

        self.decoder = CLIPSegDecoder(self.clipseg_config)

    def _forward_vit(self, x: torch.TensorType, output_hidden_states: bool = True):
        ViT = self.biomedclip_model.visual.trunk
        x = ViT.patch_embed(x)
        x = ViT._pos_embed(x)
        x = ViT.patch_drop(x)
        x = ViT.norm_pre(x)

        if output_hidden_states:
            hidden_states = (x,)

        for i, block in enumerate(ViT.blocks):
            x = block(x)
            if output_hidden_states:
                hidden_states += (x,)

        x = ViT.norm(x)

        x = ViT.forward_head(x)

        # Linear Projection: 768 -> 512
        x = self.biomedclip_model.visual.head(x)

        if output_hidden_states:
            return x, hidden_states
        else:
            return x

    def _forward_bert(self, x, output_hidden_states: bool = True):
        bert = self.biomedclip_model.text
        attn_mask = (x != bert.config.pad_token_id).long()
        out = bert.transformer(
            input_ids=x,
            attention_mask=attn_mask,
            output_hidden_states=output_hidden_states,
        )
        pooled_out = bert.pooler(out, attn_mask)
        projected = bert.proj(pooled_out)

        hidden_states = [out.hidden_states[i] for i in self.cond_layers]

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[
                :, torch.arange(seq_len) != bert.pooler.cls_token_position, :
            ]
            if type(bert.pooler) == ClsPooler
            else out.last_hidden_state
        )

        if bert.output_tokens:
            return projected, tokens

        if output_hidden_states:
            return projected, hidden_states
        else:
            return projected

    def forward(
        self, pixel_values: torch.TensorType, input_ids: torch.TensorType, **kwargs
    ):
        texts_embeds = self.biomedclip_model.text(input_ids)
        images_embeds, vit_hidden_states = self._forward_vit(pixel_values)

        # we add +1 here as the hidden states also include the initial embeddings
        activations = [
            vit_hidden_states[i + 1] for i in self.clipseg_config.extract_layers
        ]

        decoder_outputs = self.decoder(activations, texts_embeds)
        logits = decoder_outputs[0]
        if logits.ndim == 3:
            logits = logits[:, None]
        elif logits.ndim == 2:
            logits = logits[None, None]
        # Resize logits to the size of GT masks
        logits = resize(logits, [self.mask_size, self.mask_size])

        # Return mask logits
        return logits
