from torch import nn
import torch
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor, AutoTokenizer
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)


class CLIPSegCoOp(nn.Module):
    def __init__(
        self,
        clipseg_hf_api: str,
        n_ctx: int = 73,
        ctx_init: str = "",
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__()

        self.n_ctx = n_ctx

        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_hf_api)

        self.clipseg.clip.requires_grad_(not freeze_encoder)
        self.clipseg.decoder.requires_grad_(not freeze_decoder)

        tokenizer: AutoTokenizer = CLIPSegProcessor.from_pretrained(
            clipseg_hf_api
        ).tokenizer

        clipseg_token_embedder = CLIPSegForImageSegmentation.from_pretrained(
            clipseg_hf_api
        ).clip.text_model.embeddings

        # Initialize the context parameters
        ctx_token: torch.Tensor = tokenizer(ctx_init, return_tensors="pt").input_ids
        ctx_token = ctx_token[:, :-1]  # Remove the EOS token

        ctx_token = ctx_token.reshape([1, -1])
        ctx = clipseg_token_embedder(ctx_token)
        self.ctx = nn.Parameter(ctx.repeat([1, self.n_ctx, 1]), requires_grad=True)

    def text_forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        ctx = self.ctx.repeat([B, 1, 1])

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        coop_input_shape = [input_shape[0], input_shape[1] + self.n_ctx]
        hidden_states = self.clipseg.clip.text_model.embeddings(input_ids=input_ids)

        # Update the embeddings and attention mask with the context params
        hidden_states = torch.cat((hidden_states, ctx), dim=1)
        attention_mask = torch.cat(
            (
                torch.ones(
                    (B, self.n_ctx),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=1,
        )

        causal_attention_mask = _create_4d_causal_attention_mask(
            coop_input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype
            )
        encoder_outputs = self.clipseg.clip.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        last_hidden_state = encoder_outputs[0]
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
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:

        B, C, H, W = pixel_values.shape
        # step 1: forward the query images through the frozen CLIP vision encoder
        vision_outputs = self.clipseg.clip.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
        )
        # pooled_output = self.clip.visual_projection(vision_outputs[1])

        vision_hidden_states = vision_outputs.hidden_states

        # we add +1 here as the hidden states also include the initial embeddings

        vision_activations = [
            vision_hidden_states[i + 1] for i in self.clipseg.extract_layers
        ]

        conditional_embeddings = self.text_forward(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # step 4: forward both the hidden_activations and fused embedding through the lightweight decoder to predict masks
        decoder_outputs = self.clipseg.decoder(
            vision_activations,
            conditional_embeddings,
            output_hidden_states=True,
        )
        logits = decoder_outputs.logits

        return logits.view(B, 1, H, W)
