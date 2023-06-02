import json
import os
import sys
from pathlib import Path
from string import Template
from typing import Iterable, Union

import pandas as pd

from features_from_img import mask_to_overall_bbox, get_mask_description
from PIL import Image
from tqdm import tqdm

sys.path.append("/mnt/Enterprise/miccai_2023_CRIS/vqa_dir/OFA/")

from single_inference import get_answer

StrPath = Union[str, Path]


def convert_str_to_template(temp_str: Union[Iterable[str], str]):
    if isinstance(temp_str, str):
        return Template(temp_str)

    return tuple(map(Template, temp_str))


observations = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

mask_index = [
    "airspace_opacity",
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged_cardiomediastinum",
    "lung_lesion",
    "pleural_effusion",
    "pneumothorax",
    "support_devices",
]

p0 = ""

p1_template = convert_str_to_template("Chest Xray, with $labels")

p2_template = convert_str_to_template(
    "Chest Xray, $xray_view view, with $labels",
)

p3_template = convert_str_to_template(
    "Chest Xray, $xray_view view, with $labels, of shape $shape"
)

p4_template = convert_str_to_template(
    "Chest Xray, $xray_view view, with $labels, of shape $shape, and located in $location of the image"
)


def get_image_mask_paths(img_dir, mask_dir, mask_name):
    img_dir = Path(img_dir)
    image_paths = list(img_dir.glob("*.jpg"))
    assert len(image_paths) > 0, "No images found in the image directory."

    mask_dir = Path(mask_dir)
    mask_paths = list(mask_dir.glob(f"*_{mask_name}*.png"))
    assert len(mask_paths) > 0, "No files found in the mask directory."

    image_mask_paths = tuple()
    for image_path in image_paths:
        mask_path = mask_dir.joinpath(image_path.stem + f"_{mask_name}.png")
        if mask_path.exists():
            image_mask_paths += ((image_path, mask_path),)
    return image_mask_paths


def get_json_data(
    img_dir: StrPath,
    mask_dir: StrPath,
):
    all_image_mask_paths = tuple(
        get_image_mask_paths(img_dir, mask_dir, mask_name) for mask_name in mask_index
    )

    # convert all_image_mask_paths to a single list
    all_image_mask_paths = [
        image_mask_path
        for image_mask_paths in all_image_mask_paths
        for image_mask_path in image_mask_paths
    ]

    img_mask_prompt_json = []

    vqa_questions_answers = pd.DataFrame(
        columns=[
            "question",
            "answer",
            "image_path",
            "mask_path",
            "mask",
        ]
    )

    for i, image_mask_path in enumerate(tqdm(all_image_mask_paths)):
        image_path, mask_path = image_mask_path

        view, mask_name = mask_path.stem.split("_", 4)[-2:]

        p1 = p1_template.substitute(labels=mask_name)

        p2 = p2_template.substitute(xray_view=view, labels=mask_name)

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            shape = get_answer(
                image,
                mask,
                question=f"What is the shape of {mask_name} enclosed in green box?",
                verbose=True,
            )

        vqa_questions_answers.loc[i] = [
            f"What is the shape of {mask_name} enclosed in green box?",
            shape,
            image_path,
            mask_path,
            mask_name,
        ]

        p3 = p3_template.substitute(xray_view=view, labels=mask_name, shape=shape)

        location = get_mask_description(str(mask_path))[1]

        p4 = p4_template.substitute(
            xray_view=view, labels=mask_name, shape=shape, location=location
        )

        # p5 = [
        #     temp.substitute(
        #         num_chambers=num_chambers,
        #         cycle=cycle,
        #         age=age,
        #         gender=gender,
        #         image_quality=image_quality,
        #     )
        #     for temp in p5_templates
        # ]

        bbox = mask_to_overall_bbox(str(mask_path))

        # mask_name = mask_path.stem + f"_{label_index}" + mask_path.suffix

        img_name = image_path.name

        sent = [{"idx": 0, "sent_id": i, "sent": ""}]
        op = {
            "bbox": bbox,
            "cat": 0,
            "segment_id": image_path.stem,
            "img_name": img_name,
            "mask_name": mask_path.name,
            "sentences": sent,
            "sentences_num": 1,
            "prompts": {
                "p0": p0,
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                # "p5": p5,
            },
        }

        img_mask_prompt_json.append(op)
    return img_mask_prompt_json, vqa_questions_answers


def get_splits_json(base_dir: StrPath, output_dir: StrPath):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)

    val_img_dir = base_dir.joinpath("val_images")
    val_mask_dir = base_dir.joinpath("val_masks")

    test_img_dir = base_dir.joinpath("test_images")
    test_mask_dir = base_dir.joinpath("test_masks")

    train_json = []
    val_json, vqa_questions_answers_val = get_json_data(val_img_dir, val_mask_dir)
    test_json, vqa_questions_answers_test = get_json_data(test_img_dir, test_mask_dir)

    os.makedirs(output_dir, exist_ok=True)

    train_json_path = output_dir.joinpath("train.json")
    val_json_path = output_dir.joinpath("val.json")
    test_json_path_A = output_dir.joinpath("testA.json")
    test_json_path_B = output_dir.joinpath("testB.json")

    with open(train_json_path, "w") as f:
        json.dump(train_json, f)

    with open(val_json_path, "w") as f:
        json.dump(val_json, f)

    with open(test_json_path_A, "w") as f:
        json.dump(test_json, f)

    with open(test_json_path_B, "w") as f:
        json.dump(test_json, f)

    vqa_questions_answers_val.to_csv(
        output_dir.joinpath("vqa_questions_answers_val.csv"), index=False
    )
    vqa_questions_answers_test.to_csv(
        output_dir.joinpath("vqa_questions_answers_test.csv"), index=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    print(args)
    get_splits_json(**vars(args))

    print("Annotations Created")
