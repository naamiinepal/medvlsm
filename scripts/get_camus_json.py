import concurrent.futures
import json
import random
from pathlib import Path
from string import Template
from typing import Any, Iterable, Optional, Tuple, Union
from collections import defaultdict

from features_from_img import mask_to_overall_bbox
from num2words import num2words
from tqdm import tqdm
from utils import assert_split_ratio, get_train_val_test

StrPath = Union[str, Path]


def convert_str_to_template(temp_str: Union[Iterable[str], str]):
    """Convert a string or an iterable of strings to a template

    Args:
        temp_str (Union[Iterable[str], str]): The string or iterable of strings to convert

    Returns:
        Union[Template, Tuple[Template]]: The template or tuple of templates
    """

    if isinstance(temp_str, str):
        return Template(temp_str)

    return tuple(map(Template, temp_str))


mask_index_to_parts = ["Left ventricular cavity", "Myocardium", "Left atrium cavity"]

p0 = ""

p1_templates = convert_str_to_template(
    (
        "$label_name of the heart.",
        "$label_name in the cardiac ultrasound.",
    )
)


p2_templates = convert_str_to_template(
    (
        "$label_name in $num_chambers-chamber view of the heart.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound.",
    )
)

p3_templates = convert_str_to_template(
    (
        "$label_name in $num_chambers-chamber view of the heart at end of the $cycle cycle.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle.",
    )
)

p4_templates = convert_str_to_template(
    (
        "$label_name in $num_chambers-chamber view of the heart at end of the $cycle cycle of a $gender.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle of a $gender.",
    )
)

p5_templates = convert_str_to_template(
    (
        "$label_name in $num_chambers-chamber view of the heart at end of the $cycle cycle of a $age-year-old $gender.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle of a $age-year-old $gender.",
    )
)

p6_templates = convert_str_to_template(
    (
        "$label_name in $num_chambers-chamber view of the heart at end of the $cycle cycle of a $age-year-old $gender with $image_quality image quality.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle of a $age-year-old $gender with $image_quality image quality.",
    )
)


def get_single_json(
    default_prompt: str,
    raw_root: Path,
    image_index: int,
    image_mask_path: Tuple[Path, Path],
):
    """Get the json for a single image-mask pair

    Args:
        default_prompt (str): The default prompt for the image
        raw_root (Path): The path to the raw data
        image_index (int): The index of the image
        image_mask_path (Tuple[Path, Path]): The tuple of the paths to the image and mask

    Raises:
        ValueError: The image and mask names differ

    Returns:
        dict: The json for containing the bbox, prompts and sentences
    """

    image_path, mask_path = image_mask_path

    assert (
        image_path.stem + "_gt" == mask_path.stem
    ), f"Image and mask names differ at index: {image_index}. Image path: {image_path}, mask path: {mask_path}."

    mask_path_parent_name = mask_path.parent.name
    try:
        label_index = int(mask_path_parent_name[-1])
    except ValueError:
        raise ValueError(
            f"Mask directory name should end with a number for {mask_path}."
        )

    # Get minimum of 1 and maximum of 3
    label_index = min(max(label_index, 1), 3)

    # Subtract to get the index in the list
    label_name = mask_index_to_parts[label_index - 1]

    temp_sub_kwargs = {"label_name": label_name}
    p1 = [temp.substitute(**temp_sub_kwargs) for temp in p1_templates]

    mask_path_stem = mask_path.stem

    patient_id, _chamber, _stage, *_ = mask_path_stem.split("_")

    num_chambers = "two" if _chamber == "2CH" else "four"

    temp_sub_kwargs["num_chambers"] = num_chambers
    p2 = [temp.substitute(**temp_sub_kwargs) for temp in p2_templates]

    cycle = "systole" if _stage == "ES" else "diastole"

    temp_sub_kwargs["cycle"] = cycle
    p3 = [temp.substitute(**temp_sub_kwargs) for temp in p3_templates]

    with open(raw_root / patient_id / f"Info_{_chamber}.cfg") as f:
        content = f.read()

        # Split into lines
    content = content.splitlines()

    # Separate by colon
    key_value_tuple = (line.split(":", 1) for line in content)

    # Remove leading and trailing spaces
    key_value_mapping = {key.strip(): value.strip() for key, value in key_value_tuple}

    _gender = key_value_mapping["Sex"]

    gender = "female" if _gender == "F" else "male"

    temp_sub_kwargs["gender"] = gender
    p4 = [temp.substitute(**temp_sub_kwargs) for temp in p4_templates]

    age = num2words(key_value_mapping["Age"])

    temp_sub_kwargs["age"] = age
    p5 = [temp.substitute(**temp_sub_kwargs) for temp in p5_templates]

    image_quality = key_value_mapping["ImageQuality"].lower()

    temp_sub_kwargs["image_quality"] = image_quality
    p6 = [temp.substitute(**temp_sub_kwargs) for temp in p6_templates]

    bbox = mask_to_overall_bbox(str(mask_path))

    mask_name = mask_path.stem + f"_{label_index}" + mask_path.suffix

    img_name = image_path.name

    return {
        "bbox": bbox,
        "cat": 0,
        "segment_id": image_path.stem,
        "img_name": img_name,
        "mask_name": mask_name,
        "sentences": [{"idx": 0, "sent_id": image_index, "sent": default_prompt}],
        "sentences_num": 1,
        "prompts": {
            "p0": p0,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "p5": p5,
            "p6": p6,
        },
    }


def get_json_data(
    img_dir: StrPath,
    img_pattern: str,
    mask_dir: StrPath,
    mask_pattern: str,
    output_dir: StrPath,
    val_ratio: float,
    test_ratio: float,
    default_prompt: str,
    max_workers: Optional[int],
    seed: Union[int, float, str, bytes, bytearray, None],
):
    """Get the json data for the CAMUS dataset

    Args:
        img_dir (StrPath): The path to the image directory
        img_pattern (str): The glob pattern for the images
        mask_dir (StrPath): The path to the mask directory
        mask_pattern (str): The glob pattern for the masks
        output_dir (StrPath): The path to the output directory
        val_ratio (float): The ratio of validation data
        test_ratio (float): The ratio of test data
        default_prompt (str): The default prompt for the images
        max_workers (Optional[int]): The max workers for multiprocessing
        seed (Union[int, float, str, bytes, bytearray, None]): The seed for random choices
    """

    assert_split_ratio(val_ratio=val_ratio, test_ratio=test_ratio)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(img_dir)
    image_paths = list(img_dir.glob(img_pattern))
    assert len(image_paths) > 0, "No images found in the image directory."

    mask_dir = Path(mask_dir)
    mask_paths = list(mask_dir.glob(mask_pattern))
    assert len(mask_paths) > 0, "No files found in the mask directory."

    assert (
        len(mask_paths) % len(image_paths) == 0
    ), "The number of masks should be the multiple of the number of images."

    assert len(image_paths) == len(
        set(path.name for path in mask_paths)
    ), "The number of images and the number of masks should be equal. "

    # Sort image and mask paths
    image_paths.sort()

    # Sort only by the last names
    mask_paths.sort(key=lambda x: x.name)

    mask_multiple = len(mask_paths) // len(image_paths)

    image_paths = [path for path in image_paths for _ in range(mask_multiple)]

    assert len(image_paths) == len(
        mask_paths
    ), "The number of images and masks differ even after adjusting them."

    image_mask_paths = tuple(zip(image_paths, mask_paths))

    # Split on patient level
    patient_ids = set(map(get_patient_id_from_image_mask_path, image_mask_paths))

    # Set the random seed
    random.seed(seed)

    _, val_patient_ids, test_patient_ids = get_train_val_test(
        patient_ids, val_ratio, test_ratio
    )

    val_patient_ids = set(val_patient_ids)
    test_patient_ids = set(test_patient_ids)

    raw_root = Path("/mnt/Enterprise/PUBLIC_DATASETS/camus_database/training")

    train_json = []
    val_json = []
    test_json = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures_to_image_mask_path = {}
        for i, image_mask_path in enumerate(image_mask_paths, 1):
            future = executor.submit(
                get_single_json, default_prompt, raw_root, i, image_mask_path
            )
            futures_to_image_mask_path[future] = image_mask_path

        for future in tqdm(
            concurrent.futures.as_completed(futures_to_image_mask_path),
            total=len(futures_to_image_mask_path),
        ):
            image_mask_path = futures_to_image_mask_path[future]

            try:
                op = future.result()
            except Exception as e:
                print(f"Image path: {image_mask_path}")
                print(f"Exception: {e}")
            else:
                patient_id = get_patient_id_from_image_mask_path(image_mask_path)

                if patient_id in val_patient_ids:
                    val_json.append(op)
                elif patient_id in test_patient_ids:
                    test_json.append(op)
                else:  # for train data
                    train_json.append(op)

    print("\n\nLengths", len(train_json), len(val_json), len(test_json))

    print(
        "Expected Proportion",
        1 - val_ratio - test_ratio,
        val_ratio,
        test_ratio,
    )

    print(
        "Actual Proportion",
        len(train_json) / len(image_mask_paths),
        len(val_json) / len(image_mask_paths),
        len(test_json) / len(image_mask_paths),
    )

    with open(output_dir / "train.json", "w") as of:
        json.dump(train_json, of)

    with open(output_dir / "val.json", "w") as of:
        json.dump(val_json, of)

    with open(output_dir / "testA.json", "w") as of:
        json.dump(test_json, of)

    with open(output_dir / "testB.json", "w") as of:
        json.dump(test_json, of)


def get_patient_id_from_image_mask_path(image_mask_path: Tuple[Path, Any]):
    image_path, _ = image_mask_path

    patient_id = image_path.stem.split("_", 1)[0]
    return patient_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--img-pattern", type=str, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--mask-pattern", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, required=True)
    parser.add_argument("--test-ratio", type=float, required=True)
    parser.add_argument("--default-prompt", type=str, default="heart ultrasound")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=444)

    args = parser.parse_args()

    get_json_data(**vars(args))
