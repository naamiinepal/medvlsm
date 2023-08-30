import concurrent.futures
import json
import os
import random
import sys
from pathlib import Path
from string import Template
from typing import Any, Iterable, Tuple, Union

from features_from_img import get_mask_description, mask_to_overall_bbox
from PIL import Image
from tqdm import tqdm
from scripts.utils import assert_split_ratio, get_train_val_test

sys.path.append("/mnt/Enterprise/kanchan/VLM-SEG-2023/OFA/")

from single_img_inference import get_answer, return_model

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

p1_template = convert_str_to_template("$labels in a Chest Xray.")

p2_template = convert_str_to_template(
    "$labels in the $xray_view view of a Chest Xray.",
)

p3_template = convert_str_to_template(
    "$labels of shape $shape in the $xray_view view of a Chest Xray."
)

p4_template = convert_str_to_template(
    "$labels of shape $shape, and located in $location of the $xray_view view of a Chest Xray."
)


def get_json_data(
    image_dir: StrPath,
    mask_dir: StrPath,
    output_dir: StrPath,
    test_ratio: float,
    val_ratio: float,
    max_workers: int,
    seed: int = 42,
    verbose: bool = True,
):
    assert_split_ratio(val_ratio=val_ratio, test_ratio=test_ratio)

    os.makedirs(output_dir, exist_ok=True)

    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob("*.jpg"))
    assert len(image_paths) > 0, "No images found in the image directory."

    mask_dir = Path(mask_dir)

    image_mask_paths = []
    for image_path in image_paths:
        for mask_name in mask_index:
            mask_path = mask_dir.joinpath(image_path.stem + f"_{mask_name}.png")
            if mask_path.exists():
                image_mask_paths += ((image_path, mask_path),)

    assert len(image_mask_paths) > 0, "No masks found in the mask directory."

    # Split on patient level
    patient_ids = set(map(get_patient_id_from_image_mask_path, image_mask_paths))

    # set the random seed
    random.seed(seed)

    _, val_patient_ids, test_patient_ids = get_train_val_test(
        patient_ids, val_ratio, test_ratio
    )

    val_patient_ids = set(val_patient_ids)
    test_patient_ids = set(test_patient_ids)

    train_image_mask_paths = []
    val_image_mask_paths = []
    test_image_mask_paths = []

    for image_mask_path in image_mask_paths:
        patient_id = get_patient_id_from_image_mask_path(image_mask_path)
        if patient_id in val_patient_ids:
            val_image_mask_paths += (image_mask_path,)
        elif patient_id in test_patient_ids:
            test_image_mask_paths += (image_mask_path,)
        else:
            train_image_mask_paths += (image_mask_path,)

    train_json = []
    val_json = []
    test_json = []

    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures_to_image_mask_path = {}
    #     for image_mask_path in image_mask_paths:
    #         future = executor.submit(get_single_json, image_mask_path=image_mask_path)
    #         futures_to_image_mask_path[future] = image_mask_path

    #     for future in tqdm(
    #         concurrent.futures.as_completed(futures_to_image_mask_path),
    #         total=len(futures_to_image_mask_path),
    #     ):
    #         image_mask_path = futures_to_image_mask_path[future]

    #         try:
    #             img_mask_prompt_json = future.result()
    #         except Exception as exc:
    #             print(f"{image_mask_path} generated an exception: {exc}")
    #         else:
    #             patient_id = get_patient_id_from_image_mask_path(image_mask_path)
    #             if patient_id in val_patient_ids:
    #                 val_json.append(img_mask_prompt_json)
    #             elif patient_id in test_patient_ids:
    #                 test_json.append(img_mask_prompt_json)
    #             else:
    #                 train_json.append(img_mask_prompt_json)

    for idx, image_mask_path in enumerate(
        tqdm(image_mask_paths, total=len(image_mask_paths))
    ):
        img_mask_prompt_json = get_single_json(image_mask_path=image_mask_path, idx=idx)
        patient_id = get_patient_id_from_image_mask_path(image_mask_path)
        if patient_id in val_patient_ids:
            val_json.append(img_mask_prompt_json)
        elif patient_id in test_patient_ids:
            test_json.append(img_mask_prompt_json)
        else:
            train_json.append(img_mask_prompt_json)

    if verbose:
        print("Train, Val, Test")
        print(
            len(train_image_mask_paths),
            len(val_image_mask_paths),
            len(test_image_mask_paths),
        )

        # find occurrence of each of item in mask_index in the train, val and test set
        train_occurrence = {mask_name: 0 for mask_name in mask_index}
        val_occurrence = {mask_name: 0 for mask_name in mask_index}
        test_occurrence = {mask_name: 0 for mask_name in mask_index}

        for image_mask_path in train_image_mask_paths:
            for mask_name in mask_index:
                mask_path = image_mask_path[1]
                if mask_name in mask_path.stem:
                    train_occurrence[mask_name] += 1

        for image_mask_path in val_image_mask_paths:
            for mask_name in mask_index:
                mask_path = image_mask_path[1]
                if mask_name in mask_path.stem:
                    val_occurrence[mask_name] += 1

        for image_mask_path in test_image_mask_paths:
            for mask_name in mask_index:
                mask_path = image_mask_path[1]
                if mask_name in mask_path.stem:
                    test_occurrence[mask_name] += 1

        # also find if any image path is in more than one set
        train_image_paths = set(map(lambda x: x[0], train_image_mask_paths))
        val_image_paths = set(map(lambda x: x[0], val_image_mask_paths))
        test_image_paths = set(map(lambda x: x[0], test_image_mask_paths))

        assert (
            len(train_image_paths.intersection(val_image_paths)) == 0
        ), "Some image paths are in both train and val set"

        assert (
            len(train_image_paths.intersection(test_image_paths)) == 0
        ), "Some image paths are in both train and test set"

        assert (
            len(val_image_paths.intersection(test_image_paths)) == 0
        ), "Some image paths are in both val and test set"

        print(train_occurrence)
        print(val_occurrence)
        print(test_occurrence)

    with open(output_dir / "train.json", "w") as f:
        json.dump(train_json, f)

    with open(output_dir / "val.json", "w") as f:
        json.dump(val_json, f)

    with open(output_dir / "testA.json", "w") as f:
        json.dump(test_json, f)

    with open(output_dir / "testB.json", "w") as f:
        json.dump(test_json, f)


def get_patient_id_from_image_mask_path(image_mask_path: Tuple[Path, Any]):
    image_path, _ = image_mask_path
    patient_id = image_path.stem.split("_")[0]
    return patient_id


def get_single_json(image_mask_path: Tuple[Path, Path], idx: int):
    """
    Get json for a single image-mask pair

    Parameters
    ----------
    image_mask_path : Tuple[Path, Path]
        [Tuple of paths of image and mask]

    Returns
    -------
    img_mask_prompt_json : Dict
        [Json with image, mask and prompts]
    """
    image_path, mask_path = image_mask_path

    view, mask_name = mask_path.stem.split("_", 4)[-2:]

    p1 = p1_template.substitute(labels=mask_name)

    p2 = p2_template.substitute(xray_view=view, labels=mask_name)

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        shape = get_answer(
            image,
            mask,
            question=f"What is the shape of {mask_name} enclosed in green box?",
            verbose=False,
        )

    p3 = p3_template.substitute(xray_view=view, labels=mask_name, shape=shape)

    location = get_mask_description(str(mask_path))[1]

    p4 = p4_template.substitute(
        xray_view=view, labels=mask_name, shape=shape, location=location
    )

    bbox = mask_to_overall_bbox(str(mask_path))

    img_name = image_path.name

    sent = [{"idx": 0, "sent_id": idx, "sent": ""}]
    return {
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
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-dir", type=Path, required=True)
    parser.add_argument("-m", "--mask-dir", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("-t", "--test-ratio", type=float, default=0.2)
    parser.add_argument("-v", "--val-ratio", type=float, default=0.2)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)

    args = parser.parse_args()

    print(args)
    # get_splits_json(**vars(args))
    get_json_data(**vars(args))

    print("Annotations Created")
