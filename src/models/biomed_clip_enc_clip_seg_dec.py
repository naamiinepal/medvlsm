# from typing import Optional

# import open_clip
# import torch
# from open_clip.hf_model import ClsPooler
# from torch import nn
# from transformers import CLIPSegConfig, CLIPSegForImageSegmentation


# class BiomedCLIPEncCLIPSegDec(nn.Module):
#     """
#     BiomedCLIP Encoder + CLIPSeg Decoder
#     """

#     def __init__(
#         self,
#         biomedclip_hf_api: str,
#         clip_seg_hf_api: str,
#         freeze_encoder=True,
#         freeze_decoder=True,
#         rand_init_decoder=False,
#     ):
#         super().__init__()
#         # Encoder from BiomedCLIP
#         self.biomedclip = open_clip.create_model(biomedclip_hf_api)

#         self.clip_seg_config = CLIPSegConfig.from_pretrained(clip_seg_hf_api)

#         # Randomly initialize decoder
#         if rand_init_decoder:
#             self.decoder = CLIPSegForImageSegmentation(self.clip_seg_config).decoder

#         # Use pretrained decoder
#         else:
#             self.decoder = CLIPSegForImageSegmentation.from_pretrained(
#                 clip_seg_hf_api
#             ).decoder

#         if freeze_encoder:
#             for p in self.biomedclip.parameters():
#                 p.requires_grad = False

#         if freeze_decoder:
#             for p in self.decoder.parameters():
#                 p.requires_grad = False

#     def _forward_vit(self, x, output_hidden_states: bool = True):
#         ViT = self.biomedclip.visual.trunk
#         x = ViT.patch_embed(x)
#         x = ViT._pos_embed(x)
#         x = ViT.norm_pre(x)

#         hidden_states = []

#         for i, block in enumerate(ViT.blocks):
#             x = block(x)

#             hidden_states.append(x)

#         x = ViT.norm(x)

#         if ViT.global_pool:
#             x = (
#                 x[:, ViT.num_prefix_tokens :].mean(dim=1)
#                 if ViT.global_pool == "avg"
#                 else x[:, 0]
#             )
#         x = ViT.fc_norm(x)
#         x = ViT.head(x)

#         # Linear Projection: 768 -> 512
#         x = self.biomedclip.visual.head(x)

#         if output_hidden_states:
#             return x, hidden_states
#         else:
#             return x

#     def _forward_bert(
#         self,
#         x,
#         attention_mask: Optional[torch.LongTensor] = None,
#         output_hidden_states: bool = False,
#     ):
#         bert = self.biomedclip.text

#         if attention_mask is None:
#             attention_mask = (x != bert.config.pad_token_id).long()

#         out = bert.transformer(
#             input_ids=x,
#             attention_mask=attention_mask,
#             output_hidden_states=output_hidden_states,
#         )
#         pooled_out = bert.pooler(out, attention_mask)
#         projected = bert.proj(pooled_out)

#         seq_len = out.last_hidden_state.shape[1]
#         tokens = (
#             out.last_hidden_state[
#                 :, torch.arange(seq_len) != bert.pooler.cls_token_position, :
#             ]
#             if type(bert.pooler) == ClsPooler
#             else out.last_hidden_state
#         )

#         if bert.output_tokens:
#             return projected, tokens

#         if output_hidden_states:
#             return projected, out.hidden_states
#         else:
#             return projected

#     def get_conditional_embeddings(
#         self,
#         batch_size: int,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#     ):
#         # compute conditional embeddings from texts
#         if len(input_ids) != batch_size:
#             raise ValueError(
#                 "Make sure to pass as many prompt texts as there are query images"
#             )
#         conditional_embeddings = self._forward_bert(
#             input_ids, attention_mask=attention_mask, output_hidden_states=False
#         )
#         return conditional_embeddings

#     def forward(
#         self,
#         pixel_values: torch.Tensor,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#     ) -> torch.Tensor:
#         # step 1: forward the query images through the frozen CLIP vision encoder
#         with torch.inference_mode():
#             pooled_output, hidden_states = self._forward_vit(
#                 pixel_values, output_hidden_states=True
#             )
#             # we add +1 here as the hidden states also include the initial embeddings
#             activations = [
#                 hidden_states[i + 1] for i in self.clip_seg_config.extract_layers
#             ]

#         # step 2: compute conditional embeddings, either from text
#         conditional_embeddings = self.get_conditional_embeddings(
#             batch_size=pixel_values.shape[0],
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )

#         # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
#         decoder_outputs = self.decoder(activations, conditional_embeddings)
#         logits = decoder_outputs.logits

#         return logits[:, None]
