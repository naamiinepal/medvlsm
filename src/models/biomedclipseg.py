import math
from typing import List

import open_clip
import torch
from open_clip.hf_model import ClsPooler
from torch import nn


class BiomedCLIPSeg(nn.Module):
    def __init__(
        self,
        extract_layers: List[int] = [4, 7, 10, 12],
        cond_layers: List[int] = [12],
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

        self.biomedclip_model = open_clip.create_model(biomedclip_hf_api)

        for p in self.biomedclip_model.parameters():
            p.requires_grad = False

        # Projections to aggregate the text embeddings
        self.film_mul = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in cond_layers])
        self.film_add = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in cond_layers])

        # Decoder parts
        self.extract_layers = extract_layers
        self.cond_layers = cond_layers
        self.reduce_dim = reduce_dim
        self.reducers = nn.ModuleList(
            [nn.Linear(768, reduce_dim) for _ in extract_layers]
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=reduce_dim, nhead=n_heads, batch_first=True
                )
                for _ in extract_layers
            ]
        )
        self.extra_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=reduce_dim, nhead=n_heads, batch_first=True
                )
                for _ in range(extra_blocks)
            ]
        )

        # Projection to generate the binary mask
        self.conv_trans = nn.ConvTranspose2d(reduce_dim, 1, 16, 16)

    def _forward_vit(self, x: torch.TensorType, output_hidden_states: bool = True):
        ViT = self.biomedclip_model.visual.trunk
        x = ViT.patch_embed(x)
        x = ViT._pos_embed(x)
        x = ViT.norm_pre(x)

        if 0 in self.extract_layers:
            hidden_states = [x]
        else:
            hidden_states = []

        for i, block in enumerate(ViT.blocks):
            x = block(x)

            if i + 1 in self.extract_layers:
                hidden_states.append(x)

        x = ViT.norm(x)

        if ViT.global_pool:
            x = (
                x[:, ViT.num_prefix_tokens :].mean(dim=1)
                if ViT.global_pool == "avg"
                else x[:, 0]
            )
        x = ViT.fc_norm(x)
        x = ViT.head(x)

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

    def forward(self, images: torch.TensorType, texts: torch.TensorType):
        images_embeds, vit_hidden_states = self._forward_vit(images)
        texts_embeds, bert_hidden_states = self._forward_bert(texts)
        vit_hidden_states = vit_hidden_states[::-1]
        bert_hidden_states = bert_hidden_states[::-1]

        a = None
        for i, (activation, block, reducer) in enumerate(
            zip(vit_hidden_states[1:], self.blocks, self.reducers)
        ):
            if a is not None:
                a = reducer(activation) + a
            else:
                a = reducer(activation)

            if i == 0 and self.text_supervision:
                for _mul, _add, hidden_state in zip(
                    self.film_mul, self.film_add, bert_hidden_states
                ):
                    pooled_state = hidden_state[:, 0]
                    a = _mul(pooled_state)[:, None].repeat(1, 197, 1) * a + _add(
                        pooled_state
                    )[:, None].repeat(1, 197, 1)

            a = block(a)

        for block in self.extra_blocks:
            a = a + block(a)

        # Discard the CLS token and (*, tokens, features) -> (*, features, tokens)
        a = a[:, 1:].permute(0, 2, 1)

        size = int(math.sqrt(a.shape[2]))

        a = a.view(-1, a.shape[1], size, size)

        a = self.conv_trans(a)

        # Return mask logits
        return a
