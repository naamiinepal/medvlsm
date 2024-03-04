# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os

import cv2
import numpy as np
import torch


def compute_multi_iou(
    y: torch.Tensor, ignore_empty: bool = True
) -> torch.Tensor:
    """Computes Intersection over Union (IoU) score metric from a batch of predictions.

    Args:
 
        y: all predictions in (b, n, c, h, w)
        include_background: whether to include IoU computation on the first channel of
            the predicted output. Defaults to True.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.

    Returns:
        IoU scores per batch and per class, (shape [batch_size, num_classes]).

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.

    """


    #TODO: Assertion that all elemnets have the same shape


    # reducing only spatial dimensions (not batch nor channels)
    num_outputs = y.shape[0]

    n_len = len(y.shape)

    reduce_axis = list(range(2, n_len-1))

    y_mul_acc = torch.ones_like(y[0])


    for i in range(num_outputs):
        y_mul_acc = y_mul_acc*y[i]

    intersection = torch.sum(y_mul_acc, dim=reduce_axis)

    y_add_acc = torch.zeros_like(y[0])

    for i in range(num_outputs):
        y_add_acc = y_add_acc+y[i]
    
    y_union = (y_add_acc>=1).int()

    union = torch.sum(y_union, dim=reduce_axis)

    return torch.where(union > 0, (intersection) / union, torch.tensor(1.0, device=y_union.device))

# seg_root_path = "output_masks/cris/zss/kvasir_polyp"
# filename = "675.png"
# pred_img_paths = [f"{seg_root_path}/{p}/{filename}" for p in os.listdir(seg_root_path)]
# pred_imgs = np.stack((cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE) for pred_img_path in pred_img_paths), axis=0)
# # threshold the images
# pred_imgs[pred_imgs > 0] = 1
# # change images to  tensor [N,B,C,H,W]
# pred_imgs = torch.from_numpy(pred_imgs)[:, None, None, ...]
# compute_multi_iou(pred_imgs)
