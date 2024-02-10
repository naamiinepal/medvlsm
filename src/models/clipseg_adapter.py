from torch import nn
from transformers import CLIPSegForImageSegmentation
import torch


class VAdapter(nn.Module):
    """ "Vision-Language Adapter Module for fusing their representations of image and text encoders in CLIP model

    Args:
        v_dim (int): The dimension of the feature encoded by the image encoder.
        l_dim (int): The dimension of the feature encoded by the text encoder.
        embed_dim (int): The dimension of the feature encoded within the adapter.
        n_hidden_layers (int): The number of hidden blocks to propagate fused features within the adapter.
        out_dim (int): The dimension of the output layer.
    """

    def __init__(
        self,
        v_dim: int,
        embed_dim: int,
        n_hidden_layers: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.v_dim = v_dim
        self.embed_dim = embed_dim

        self.v_norm = nn.LayerNorm(self.v_dim, eps=1.0e-5)


        self.proj_v = nn.Linear(self.v_dim, self.embed_dim)

        self.activation = nn.ReLU()
        self.v_hidden_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU())
                for _ in range(n_hidden_layers)
            ]
        )

        self.v_final_layer = nn.Linear(self.embed_dim, self.v_dim)

    def forward(self, v_feat: torch.Tensor) -> torch.Tensor:
        
        v_embed = self.proj_v(v_feat)

        for v_h_layer in self.v_hidden_layers:
            v_embed = v_h_layer(v_embed)

        v_embed = self.v_norm(self.v_final_layer(v_embed))

        return v_embed

class VLAdapter(nn.Module):
    """ "Vision-Language Adapter Module for fusing their representations of image and text encoders in CLIP model

    Args:
        v_dim (int): The dimension of the feature encoded by the image encoder.
        l_dim (int): The dimension of the feature encoded by the text encoder.
        embed_dim (int): The dimension of the feature encoded within the adapter.
        n_hidden_layers (int): The number of hidden blocks to propagate fused features within the adapter.
        out_dim (int): The dimension of the output layer.
    """

    def __init__(
        self,
        v_dim: int,
        l_dim: int,
        embed_dim: int,
        n_hidden_layers: int,
        out_dim: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.v_norm = nn.LayerNorm(self.out_dim, eps=1.0e-5)

        self.l_norm = nn.LayerNorm(self.out_dim, eps=1.0e-5)

        self.proj_v = nn.Linear(self.v_dim, self.embed_dim)
        self.proj_l = nn.Linear(self.l_dim, self.embed_dim)

        self.activation = nn.ReLU()
        self.v_hidden_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU())
                for _ in range(n_hidden_layers)
            ]
        )

        self.l_hidden_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU())
                for _ in range(n_hidden_layers)
            ]
        )

        self.v_final_layer = nn.Linear(self.embed_dim, self.out_dim)
        self.l_final_layer = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, v_feat: torch.Tensor, l_feat: torch.Tensor) -> torch.Tensor:
        
        v_embed = self.proj_v(v_feat)
        l_embed = self.proj_l(l_feat)

        for v_h_layer, l_h_layer in zip(self.v_hidden_layers, self.l_hidden_layers):
            v_embed = v_h_layer(v_embed)
            l_embed = l_h_layer(l_embed)

        v_embed = self.v_norm(self.v_final_layer(v_embed))
        l_embed = self.l_norm(self.l_final_layer(l_embed))

        return v_embed + l_embed


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

        # The only trainable part of the model
        self.adapter = VLAdapter(
            v_dim=768, l_dim=512, embed_dim=512, n_hidden_layers=3, out_dim=512
        )

        self.extract_adapters = nn.ModuleList([
            VAdapter(v_dim=768, embed_dim=512, n_hidden_layers=3) for _ in self.clipseg.extract_layers
        ])

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
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
        activations = [hidden_states[i + 1] for i in self.clipseg.extract_layers]

        # step 2: compute conditional embeddings from text on;y
        conditional_embeddings = self.clipseg.get_conditional_embeddings(
            batch_size=pixel_values.shape[0],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # step 3: forward both the pooled output and conditional emnbedding to an adapter module
        fused_embeddings = self.adapter(pooled_output, conditional_embeddings)

        # step 4: forward both the hidden_activations and fused embedding through the lightweight decoder to predict masks
        decoder_outputs = self.clipseg.decoder(activations, fused_embeddings)
        logits = decoder_outputs.logits

        if logits.ndim == 2:
            return logits[None, None, :]
        else:
            return logits[:, None]