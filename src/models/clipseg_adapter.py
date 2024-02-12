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
        self.vision_extract_adapters = nn.ModuleList([
            Adapter(input_dim=768, adapter_dim=512) for _ in self.clipseg.extract_layers
        ])

        self.cond_adapter = Adapter(input_dim=512, adapter_dim=256)

        self.text_extract_adapters = nn.ModuleList([
            Adapter(input_dim=512, adapter_dim=256) for _ in self.clipseg.extract_layers
        ])

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

        vision_hidden_states, vision_pooled_output = vision_outputs.hidden_states, vision_outputs.pooler_output

        # we add +1 here as the hidden states also include the initial embeddings
        activations = [adapter(vision_hidden_states[i + 1]) for i, adapter in zip(self.clipseg.extract_layers, self.vision_extract_adapters)]


        # step 2: compute conditional embeddings from text on;y
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        text_hidden_states, text_pooled_output = text_outputs.hidden_states, text_outputs.pooler_output

        conditional_embeddings = self.clip.text_projection(text_pooled_output)
        conditional_embeddings = self.cond_adapter(conditional_embeddings)

        for i, adapter in zip(self.clipseg.extract_layers, self.text_extract_adapters):
            conditional_embeddings += adapter(text_hidden_states[i+1][:,-1]) 

        # step 4: forward both the hidden_activations and fused embedding through the lightweight decoder to predict masks
        decoder_outputs = self.clipseg.decoder(activations, conditional_embeddings, output_hidden_states=True,)
        logits = decoder_outputs.logits

        if logits.ndim == 2:
            return logits[None, None, :]
        else:
            return logits[:, None]