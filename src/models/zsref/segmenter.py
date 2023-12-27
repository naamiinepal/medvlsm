# Adapted from https://github.com/Seonghoon-Yu/Zero-shot-RIS/
from torch import nn
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import transformers
from detectron2.checkpoint import DetectionCheckpointer
from typing import List, Tuple
import spacy
import clip

from .utils import default_argument_parser, setup
from .freesolo.engine.trainer import BaselineTrainer
from .freesolo.modeling.solov2 import PseudoSOLOv2


class ZSRef(nn.Module):
    """ "Zero-Shot Referring Image Segmentation with Global-Local Context Features
    Reference: https://arxiv.org/abs/2303.17811

    Args:
        freesolo_cfg_file (str): path of the configuration used in FreeSOLO mask proposal network
        pretrained_freesolo_path (str): path of the pretrained FreeSOLO model
        pixel_mean (Tuple): The mean value of the pixel used in RGB. Default is the official ImageNet pixel mean.
    """

    def __init__(
        self,
        freesolo_cfg_path: str,
        pretrained_freesolo_path: str,
        pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.args = default_argument_parser().parse_args(args=[])
        self.args.config_file = freesolo_cfg_path
        self.cfg = setup(self.args)
        self.pixel_mean = torch.Tensor(pixel_mean).reshape(1, 3, 1, 1)

        # Load pretrained mask proposal network
        self.freesolo = BaselineTrainer.build_model(self.cfg).eval()
        _ = DetectionCheckpointer(self.freesolo).resume_or_load(
            pretrained_freesolo_path, resume=True
        )

        # Load pretrained_clip_model
        self.clip_model, _ = clip.load("RN50")

        # Text parser
        self.parser = spacy.load("en_core_web_lg")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        sentence: [str],
        **kwargs,
    ):
        assert (
            len(sentence) == 1
            and len(pixel_values) == 1
            and len(input_ids) == 1
        ), f"Only batch size of 1 is accepted in Zero-Shot RIS."

        height, width, sentence = *pixel_values.shape[-2:], sentence[0]

        if pixel_values.device != self.pixel_mean.device:
            self.pixel_mean = self.pixel_mean.to(pixel_values)

        freesolo_input = dict(image=pixel_values, height=height, width=width)
        proposals = self.freesolo([freesolo_input])[0]
        proposed_masks = proposals["instances"].pred_masks
        proposed_boxes = proposals["instances"].pred_boxes

        clip_input = TF.resize(pixel_values, (224, 224))

        feature_map = self.clip_model.encode_image(clip_input)
        feature_map = feature_map / feature_map.norm(
            dim=1, keepdim=True
        )  # normalize feature map

        masks = TF.resize(proposed_masks.type(torch.float32), (feature_map.shape[2:]))
        masked_feature_map = torch.mul(feature_map, masks[:, None, :, :])
        global_visual_features = self.clip_model.visual.attnpool(masked_feature_map)

        original_img = TF.resize(pixel_values, (height, width))

        cropped_imgs = []

        for m, b in zip(proposed_masks, proposed_boxes):
            m, b = m.type(torch.uint8), b.type(torch.int)  # type 변환
            masked_img = (
                original_img * m[None, None, ...]
                + (1 - m[None, None, ...]) * self.pixel_mean
            )

            x_min, y_min, x_max, y_max = b
            h, w = y_max - y_min, x_max - x_min

            cropped_img = TF.resized_crop(
                masked_img.squeeze(0), y_min, x_min, h, w, (224, 224)
            )
            cropped_imgs.append(cropped_img)

        cropped_imgs = torch.stack(cropped_imgs, dim=0)

        # feature_map = self.clip_model.encode_image(cropped_imgs)
        # local_visual_features = self.clip_model.visual.attnpool(feature_map)

        local_visual_features = self.clip_model.encode_image(cropped_imgs, attn=True)
        local_visual_features = local_visual_features / local_visual_features.norm(
            dim=1, keepdim=True
        )  # normalize

        global_local_visual_features = (
            0.85 * global_visual_features + (1 - 0.85) * local_visual_features
        )

        global_textual_feature = self.clip_model.encode_text(input_ids)
        global_textual_feature = global_textual_feature / global_textual_feature.norm(
            dim=1, keepdim=True
        )

        doc = self.parser(sentence)

        chunks = {}
        for chunk in doc.noun_chunks:
            for i in range(chunk.start, chunk.end):
                chunks[i] = chunk

        try:
            for token in doc:
                if token.head.i == token.i:
                    root_word = token.head
            noun_phrase = chunks[root_word.i].text

        # For empty text prompt and condition where the root word gets misplaced
        except:
            noun_phrase = ""
            
        noun_phrase_token = clip.tokenize(noun_phrase).to(input_ids.device)
        noun_phrase_feature = self.clip_model.encode_text(noun_phrase_token)
        noun_phrase_feature = noun_phrase_feature / noun_phrase_feature.norm(
            dim=1, keepdim=True
        )  # normalize

        global_local_textual_feature = (
            0.5 * global_textual_feature + (1 - 0.5) * noun_phrase_feature
        )

        similarity = global_local_visual_features @ global_local_textual_feature.T
        max_index = torch.argmax(similarity)
        mask_prediction: torch.Tensor = proposed_masks[max_index]

        # Return mask as int Tensor and unsqueeze twice along the first dimension
        return mask_prediction.long()[None, None]  # shape: [1,1,H,W]
