from typing import Optional

from torch import nn
from transformers import CLIPSegForImageSegmentation
import torch
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)


class Adapter(nn.Module):

    def __init__(self, input_dim, adapter_dim, use_gelu=False):
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim

        super().__init__()
        self.fc1 = torch.nn.Linear(self.input_dim, self.adapter_dim)
        self.fc2 = torch.nn.Linear(self.adapter_dim, self.input_dim)
        if use_gelu:
            self.activation = torch.nn.GELU()
        else:
            self.activation = torch.nn.ReLU()

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))

        return x + h


class CLIPSegAdapter(nn.Module):

    def __init__(
        self,
        clipseg_hf_api: str,
        adapter_dim: int,
        freeze_clipseg: bool = True,
        adapter_in_v: bool = True,
        adapter_in_l: bool = True,
        adapter_in_cond: bool = True,
    ):
        super().__init__()

        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_hf_api)
        self.clip = self.clipseg.clip

        self.adapter_dim = adapter_dim
        self.adapter_in_v = adapter_in_v
        self.adapter_in_l = adapter_in_l
        self.adapter_in_cond = adapter_in_cond
        self.clipseg_config = self.clipseg.config

        self.clipseg.requires_grad_(not freeze_clipseg)

        # The trainable params
        if self.adapter_in_v:
            self.vision_extract_adapters = nn.ModuleList(
                [
                    Adapter(
                        input_dim=self.clipseg_config.vision_config.hidden_size,
                        adapter_dim=self.adapter_dim,
                    )
                    for _ in self.clipseg.extract_layers
                ]
            )

        if self.adapter_in_l:
            self.text_extract_adapters = nn.ModuleList(
                [
                    Adapter(
                        input_dim=self.clipseg_config.text_config.hidden_size,
                        adapter_dim=self.adapter_dim,
                    )
                    for _ in self.clipseg.extract_layers
                ]
            )

        if self.adapter_in_cond:
            self.cond_adapter = Adapter(
                input_dim=self.clipseg_config.projection_dim, adapter_dim=256
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        B, C, H, W = pixel_values.shape
        # step 1: forward the query images through the frozen CLIP vision encoder
        vision_outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
        )
        # pooled_output = self.clip.visual_projection(vision_outputs[1])

        vision_hidden_states, vision_pooled_output = (
            vision_outputs.hidden_states,
            vision_outputs.pooler_output,
        )

        # we add +1 here as the hidden states also include the initial embeddings

        vision_activations = []

        for i, extract_idx in enumerate(self.clipseg_config.extract_layers):
            h_state = vision_hidden_states[extract_idx + 1]
            if self.adapter_in_v:
                h_state = self.vision_extract_adapters[i](h_state)
            vision_activations.append(h_state)

        # step 2: compute conditional embeddings from text on;y
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        text_hidden_states, text_pooled_output = (
            text_outputs.hidden_states,
            text_outputs.pooler_output,
        )

        conditional_embeddings = self.clip.text_projection(text_pooled_output)
        if self.adapter_in_cond:
            # Apply adapter on the conditional embedding here
            conditional_embeddings = self.cond_adapter(conditional_embeddings)

        if self.adapter_in_l:
            for i, adapter in zip(
                self.clipseg.extract_layers, self.text_extract_adapters
            ):

                h_state = text_hidden_states[i + 1]

                # Extract the token sentence feature from 'eot_token', which is of the hihghest index
                h_state = h_state[
                    torch.arange(h_state.shape[0], device=h_state.device),
                    input_ids.to(dtype=torch.int, device=h_state.device).argmax(dim=-1),
                ]

                # Apply the adapter on the hidden embeddings of texts as skip connections to the condtional embeddings
                conditional_embeddings += adapter(h_state)

        # step 4: forward both the hidden_activations and fused embedding through the lightweight decoder to predict masks
        decoder_outputs = self.clipseg.decoder(
            vision_activations,
            conditional_embeddings,
            output_hidden_states=True,
        )
        logits = decoder_outputs.logits

        return logits.view(B, 1, H, W)


class CLIPSegDenseAdapter(nn.Module):

    def __init__(
        self,
        clipseg_hf_api: str,
        adapter_dim: int,
        freeze_clipseg: bool = True,
        adapter_in_v: bool = True,
        adapter_in_l: bool = True,
        adapter_in_cond: bool = True,
    ):
        super().__init__()

        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_hf_api)

        self.adapter_dim = adapter_dim
        self.adapter_in_v = adapter_in_v
        self.adapter_in_l = adapter_in_l
        self.adapter_in_cond = adapter_in_cond
        self.clipseg_config = self.clipseg.config

        self.clipseg.requires_grad_(not freeze_clipseg)

        # The trainable params
        if self.adapter_in_v:
            num_adapters = max(
                self.clipseg_config.extract_layers
            )  # In this case 9 adapters
            self.v_attn_adapters = nn.ModuleList(
                [
                    Adapter(
                        input_dim=self.clipseg_config.vision_config.hidden_size,
                        adapter_dim=self.adapter_dim,
                    )
                    for _ in range(num_adapters)
                ]
            )

            self.v_out_adapters = nn.ModuleList(
                [
                    Adapter(
                        input_dim=self.clipseg_config.vision_config.hidden_size,
                        adapter_dim=self.adapter_dim,
                    )
                    for _ in range(num_adapters)
                ]
            )

        # The trainable params
        if self.adapter_in_l:
            num_adapters = max(
                self.clipseg_config.extract_layers
            )  # In this case 9 adapters
            self.l_attn_adapters = nn.ModuleList(
                [
                    Adapter(
                        input_dim=self.clipseg_config.text_config.hidden_size,
                        adapter_dim=self.adapter_dim,
                    )
                    for _ in range(num_adapters)
                ]
            )

            self.l_out_adapters = nn.ModuleList(
                [
                    Adapter(
                        input_dim=self.clipseg_config.text_config.hidden_size,
                        adapter_dim=self.adapter_dim,
                    )
                    for _ in range(num_adapters)
                ]
            )

        if self.adapter_in_cond:
            self.cond_adapter = Adapter(
                input_dim=self.clipseg_config.projection_dim, adapter_dim=256
            )

    def vision_forward(self, pixel_values: torch.Tensor):

        clip_vision_model = self.clipseg.clip.vision_model
        encoder_state = clip_vision_model.embeddings(pixel_values)
        encoder_state = clip_vision_model.pre_layrnorm(encoder_state)

        encoder_hidden_states = ()
        for idx, encoder_layer in enumerate(
            clip_vision_model.encoder.layers,
        ):
            encoder_hidden_states = encoder_hidden_states + (encoder_state,)
            residual = encoder_state
            encoder_state = encoder_layer.layer_norm1(encoder_state)
            encoder_state, _ = encoder_layer.self_attn(encoder_state)

            # Apply adapter before residual
            if self.adapter_in_v and idx < len(self.v_attn_adapters):
                encoder_state = self.v_attn_adapters[idx](encoder_state)

            encoder_state = residual + encoder_state

            residual = encoder_state
            encoder_state = encoder_layer.layer_norm2(encoder_state)
            encoder_state = encoder_layer.mlp(encoder_state)

            # Apply adapter before residual adding
            if self.adapter_in_v and idx < len(self.v_out_adapters):
                encoder_state = self.v_out_adapters[idx](encoder_state)
            encoder_state = residual + encoder_state

        encoder_hidden_states = encoder_hidden_states + (encoder_state,)

        # TODO: The pooled output of vision encoder not used; thus commented, uncomment this if needed
        # vision_pooled_output = encoder_state[:, 0, :]
        # vision_pooled_output = clip_vision_model.post_layernorm(
        #     vision_pooled_output
        # )

        return encoder_hidden_states


    def text_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.clipseg.clip.text_model.embeddings(input_ids=input_ids)
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype
            )

        for idx, encoder_layer in enumerate(
            self.clipseg.clip.text_model.encoder.layers
        ):
            residual = hidden_states

            hidden_states = encoder_layer.layer_norm1(hidden_states)
            hidden_states, _ = encoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
            )

            # Apply adapter before residual addition
            if self.adapter_in_l and idx < len(self.l_attn_adapters):
                hidden_states = self.l_attn_adapters[idx](hidden_states)

            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = encoder_layer.layer_norm2(hidden_states)
            hidden_states = encoder_layer.mlp(hidden_states)

            # Apply adapter before residual addition
            if self.adapter_in_l and idx < len(self.l_out_adapters):
                hidden_states = self.l_out_adapters[idx](hidden_states)

            hidden_states = residual + hidden_states

        last_hidden_state = self.clipseg.clip.text_model.final_layer_norm(hidden_states)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                dim=-1
            ),
        ]
        text_features = self.clipseg.clip.text_projection(pooled_output)

        return text_features

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ):

        B, C, H, W = pixel_values.shape

        vision_hidden_states = self.vision_forward(pixel_values=pixel_values)

        vision_activations = [
            vision_hidden_states[i + 1] for i in self.clipseg_config.extract_layers
        ]

        conditional_embeddings = self.text_forward(input_ids, attention_mask)

        if self.adapter_in_cond:
            conditional_embeddings = self.cond_adapter(conditional_embeddings)

        decoder_outputs = self.clipseg.decoder(
            vision_activations,
            conditional_embeddings,
            output_hidden_states=True,
        )
        logits = decoder_outputs.logits
        return logits.view(B, 1, H, W)
