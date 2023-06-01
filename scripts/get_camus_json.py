import json
from pathlib import Path
import random
from typing import Optional, Tuple, Union, Iterable
from string import Template
from features_from_img import mask_to_overall_bbox
import concurrent.futures
from num2words import num2words

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


# ["Two chamber view of the heart"], ["Two chamber view in the cardiac ultrasound"]
# ["Two chamber view of the heart at dystole"], ["Two chamber view in the cardiac ultrasound at dystole"]
# ["Two chamber view of the heart at dystole of a female"], ["Two chamber view in the cardiac ultrasound at dystole of a female"]
# ["Two chamber view of the heart at dystole of a 40 year-old female"], ["Two chamber view in the cardiac ultrasound at dystole of a 40 year-old female"]
# ["Two chamber view of the heart at dystole of a 40 year-old female with poor image quality"],
#       ["Two chamber view in the cardiac ultrasound at dystole of a 40 year-old female with poor image quality"]

mask_index_to_parts = ["left ventricular cavity", "myocardium", "left atrium cavity"]

p0 = ""

p1_base = ("Echocardiography", "Cardiac Ultrasound")

p2_templates = convert_str_to_template(
    (
        "$num_chambers chamber view of the heart at end of $cycle cycle.",
        "$num_chambers chamber view in the cardiac ultrasound at end of $cycle cycle.",
    )
)

p3_templates = convert_str_to_template(
    (
        "$num_chambers chamber view of the heart at end of $cycle cycle of a $gender.",
        "$num_chambers chamber view in the cardiac ultrasound at end of $cycle cycle of a $gender.",
    )
)

p4_templates = convert_str_to_template(
    (
        "$num_chambers chamber view of the heart at end of $cycle cycle of a $age year-old $gender.",
        "$num_chambers chamber view in the cardiac ultrasound at end of $cycle cycle of a $age year-old $gender.",
    )
)

p5_templates = convert_str_to_template(
    (
        "$num_chambers chamber view of the heart at end of $cycle cycle of a $age year-old $gender with $image_quality image quality.",
        "$num_chambers chamber view in the cardiac ultrasound at end of $cycle cycle of a $age year-old $gender with $image_quality image quality.",
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

    p1 = [*p1_base, label_name]

    mask_path_stem = mask_path.stem

    patient_id, _chamber, _stage, *_ = mask_path_stem.split("_")

    num_chambers = "Two" if _chamber == "2CH" else "Four"
    cycle = "systole" if _stage == "ES" else "dystole"

    p2 = [
        temp.substitute(num_chambers=num_chambers, cycle=cycle) for temp in p2_templates
    ]

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

    p3 = [
        temp.substitute(num_chambers=num_chambers, cycle=cycle, gender=gender)
        for temp in p3_templates
    ]

    age = num2words(key_value_mapping["Age"])

    p4 = [
        temp.substitute(num_chambers=num_chambers, cycle=cycle, age=age, gender=gender)
        for temp in p4_templates
    ]

    image_quality = key_value_mapping["ImageQuality"].lower()

    p5 = [
        temp.substitute(
            num_chambers=num_chambers,
            cycle=cycle,
            age=age,
            gender=gender,
            image_quality=image_quality,
        )
        for temp in p5_templates
    ]

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

    assert val_ratio >= 0, "Validation set percent should be greater than 0."
    assert test_ratio >= 0, "Test set percent should be greater than 0."
    assert (
        val_ratio + test_ratio <= 1
    ), "The sum of percent of validation set and test set should be less or equal to 1"

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

    # Set the random seed
    random.seed(seed)

    val_size = int(val_ratio * len(image_mask_paths))
    val_data = set(random.sample(image_mask_paths, val_size))

    rem_data = tuple(x for x in image_mask_paths if x not in val_data)

    test_size = int(test_ratio * len(image_mask_paths))
    test_data = set(random.sample(rem_data, test_size))

    train_data = tuple(x for x in rem_data if x not in test_data)

    assert len(val_data) + len(test_data) + len(train_data) == len(
        image_mask_paths
    ), "Data split is invalid."

    raw_root = Path("/mnt/Enterprise/PUBLIC_DATASETS/camus_database/training")

    train_json = []
    val_json = []
    test_json = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures_to_image_mask_path = {}
        for i, image_mask_path in enumerate(image_mask_paths, 1):
            f = executor.submit(
                get_single_json, default_prompt, raw_root, i, image_mask_path
            )
            futures_to_image_mask_path[f] = image_mask_path

        for i, f in enumerate(
            concurrent.futures.as_completed(futures_to_image_mask_path), 1
        ):
            image_mask_path = futures_to_image_mask_path[f]

            try:
                op = f.result()
            except Exception as e:
                print(f"Exception occurred at {i}/{len(image_mask_paths)} images")
                print(f"Image path: {image_mask_path}")
                print(f"Exception: {e}")
            else:
                if image_mask_path in train_data:
                    train_json.append(op)
                elif image_mask_path in val_data:
                    val_json.append(op)
                else:  # for test data
                    test_json.append(op)

                print(f"Processed {i}/{len(image_mask_paths)} images")

    print(len(train_json), len(val_json), len(test_json))

    with open(output_dir / "train.json", "w") as of:
        json.dump(train_json, of)

    with open(output_dir / "val.json", "w") as of:
        json.dump(val_json, of)

    with open(output_dir / "testA.json", "w") as of:
        json.dump(test_json, of)

    with open(output_dir / "testB.json", "w") as of:
        json.dump(test_json, of)


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

    print(args, end="\n\n")
    get_json_data(**vars(args))

    print("Annotations Created")
