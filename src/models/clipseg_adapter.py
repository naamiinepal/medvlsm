from typing import Optional

from torch import nn
from transformers import CLIPSegForImageSegmentation
import torch

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
        freeze_clipseg: bool = True,
    ):
        super().__init__()

        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_hf_api)
        self.clip = self.clipseg.clip

        self.clipseg.requires_grad_(not freeze_clipseg)

        # The trainable params
        self.extract_adapters = nn.ModuleList([
            Adapter(input_dim=768, adapter_dim=512) for _ in self.clipseg.extract_layers
        ])

        self.cond_adapter = Adapter(input_dim=512, adapter_dim=512)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # step 1: forward the query images through the frozen CLIP vision encoder
        vision_outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
        )
        # pooled_output = self.clip.visual_projection(vision_outputs[1])

        hidden_states, pooled_output = vision_outputs.hidden_states, vision_outputs.pooler_output

        # we add +1 here as the hidden states also include the initial embeddings
        activations = [adapter(hidden_states[i + 1]) for i, adapter in zip(self.clipseg.extract_layers, self.extract_adapters)]

        
        # step 2: compute conditional embeddings from text on;y
        conditional_embeddings = self.clipseg.get_conditional_embeddings(
            batch_size=pixel_values.shape[0],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        conditional_embeddings = self.cond_adapter(conditional_embeddings)

        # step 3: forward both the pooled output and conditional emnbedding to an adapter module
        # fused_embeddings = self.adapter(pooled_output, conditional_embeddings)

        # step 4: forward both the hidden_activations and fused embedding through the lightweight decoder to predict masks
        decoder_outputs = self.clipseg.decoder(activations, conditional_embeddings)
        logits = decoder_outputs.logits

        if logits.ndim == 2:
            return logits[None, None, :]
        else:
            return logits[:, None]