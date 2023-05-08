# A copy of https://github.com/rajpurkarlab/cheXlocalize/blob/master/annotation_to_segmentation.py
# Modified to obtain the masks of CheXlocalize dataset as png images.

"""
This script converts the raw human annotations from the CheXlocalize dataset
into segmentation masks. The masks are saved as png images in the
corresponding task directories.

Usage:
    python json_to_mask.py --input_path <path_to_json_file> --output_path <path_to_output_dir>
"""

import argparse
import json
import os
from typing import Iterable, Sequence, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

LOCALIZATION_TASKS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Support Devices",
]


def create_mask(polygons: Iterable[Iterable[Sequence[int]]], img_dims: Sequence[int]):
    """
    Creates a binary mask (of the original matrix size) given a list of polygon
        annotations format.

    Args:
        polygons (list): [[[x11,y11],[x12,y12],...[x1n,y1n]],...]

    Returns:
        mask (np.array): binary mask, 1 where the pixel is predicted to be the,
                                                 pathology, 0 otherwise
    """
    with Image.new("1", (img_dims[1], img_dims[0])) as poly:
        for polygon in polygons:
            coords = [(point[0], point[1]) for point in polygon]
            ImageDraw.Draw(poly).polygon(coords, outline=1, fill=1)

        binary_mask = np.array(poly, dtype=int)
    return binary_mask


def ann_to_mask(
    input_path: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]],
    output_path: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]],
):
    """
    Args:
        input_path (string): json file path with raw human annotations
        output_path (string): directory path for saving segmentation masks
    """
    print(f"Reading annotations from {input_path}...")
    with open(input_path) as f:
        ann = json.load(f)

    print(f"Creating and encoding segmentations...")
    for img_id in tqdm(ann.keys()):
        for task in LOCALIZATION_TASKS:
            if task in ann[img_id].keys():
                task_dir = os.path.join(output_path, task.lower().replace(" ", "_"))
                os.makedirs(task_dir, exist_ok=True)
                # create segmentation
                polygons = ann[img_id][task] if task in ann[img_id] else []
                img_dims = ann[img_id]["img_size"]
                segm_map = create_mask(polygons, img_dims)
                segm_map = np.asarray(segm_map, dtype=np.uint8)
                segm_map[segm_map == 1] = 255
                outfile = os.path.join(task_dir, img_id + ".png")
                cv2.imwrite(outfile, segm_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the json file with the raw human annotations",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the directory for saving segmentation masks",
    )
    args = parser.parse_args()

    ann_to_mask(args.input_path, args.output_path)
