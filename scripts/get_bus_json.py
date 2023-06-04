import concurrent.futures
import json
import random
from pathlib import Path
from string import Template
from typing import Iterable, Optional, Tuple, Union

import cv2
from features_from_img import get_mask_decription, mask_to_overall_bbox
from num2words import num2words
from tqdm import tqdm
from utils import assert_split_ratio, get_train_val_test

StrPath = Union[str, Path]

p0 = ""

p2_template = Template("$tumor_type tumor in the breast ultrasound image.")

generic_template = Template("$tumor_prefix in the breast ultrasound image.")

p5_template = Template("$tumor_prefix at the $position in the breast ultrasound image.")


def capitalize_iterable(string_iterable: Iterable[str]):
    return [s.capitalize() for s in string_iterable]


def get_single_json(
    default_prompt: str,
    image_index: int,
    image_mask_path: Tuple[Path, Path],
):
    """Get the json for a single image-mask pair

    Args:
        default_prompt (str): The default prompt for the image
        image_index (int): The index of the image
        image_mask_path (Tuple[Path, Path]): The tuple of the paths to the image and mask

    Returns:
        dict: The json for containing the bbox, prompts and sentences
    """

    image_path, mask_path = image_mask_path

    assert image_path.stem == mask_path.stem, "Image and mask names do not match"

    us_type = image_path.stem.rsplit("_", 1)[-1]

    if us_type == "normal":
        p5 = p4 = p3 = p2 = p1 = generic_template.substitute(
            tumor_prefix="No tumor"
        ).capitalize()
    else:
        p1 = generic_template.substitute(tumor_prefix="tumor")

        regularity = "regular-shaped" if us_type == "benign" else "irregular-shaped"

        p2 = [
            p2_template.substitute(tumor_type=us_type),
            p2_template.substitute(tumor_type=regularity),
        ]

        mask_ori = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # sizes, positions, numbers
        size, position, number = get_mask_decription(
            mask_ori, convert_num_to_words=False
        )

        assert number > 0, f"No tumor found in the {us_type} case."

        if number > 1:
            word_num = num2words(number)

            p3 = [
                generic_template.substitute(
                    tumor_prefix=f"{word_num} {us_type} tumors"
                ),
                generic_template.substitute(
                    tumor_prefix=f"{word_num} {regularity} tumors"
                ),
            ]

            p4 = [
                generic_template.substitute(
                    tumor_prefix=f"{word_num} {size} {us_type} tumors"
                ),
                generic_template.substitute(
                    tumor_prefix=f"{word_num} {size} {regularity} tumors"
                ),
            ]

            p5 = [
                p5_template.substitute(
                    tumor_prefix=f"{word_num} {size} {us_type} tumors",
                    position=position,
                ),
                p5_template.substitute(
                    tumor_prefix=f"{word_num} {size} {regularity} tumors",
                    position=position,
                ),
            ]
        else:
            p3 = [
                generic_template.substitute(tumor_prefix=f"One {us_type} tumor"),
                generic_template.substitute(tumor_prefix=f"One {regularity} tumor"),
            ]

            p4 = [
                generic_template.substitute(tumor_prefix=f"One {size} {us_type} tumor"),
                generic_template.substitute(
                    tumor_prefix=f"One {size} {regularity} tumor"
                ),
            ]

            p5 = [
                p5_template.substitute(
                    tumor_prefix=f"One {size} {us_type} tumor", position=position
                ),
                p5_template.substitute(
                    tumor_prefix=f"One {size} {regularity} tumor", position=position
                ),
            ]

        # Capitalize the prompts
        p1 = capitalize_iterable(p1)
        p2 = capitalize_iterable(p2)
        p3 = capitalize_iterable(p3)
        p4 = capitalize_iterable(p4)
        p5 = capitalize_iterable(p5)

    bbox = mask_to_overall_bbox(str(mask_path))

    return {
        "bbox": bbox,
        "cat": 0,
        "segment_id": image_path.stem,
        "img_name": image_path.name,
        "mask_name": mask_path.name,
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


def main(
    root_dir: Path,
    image_glob: str,
    masks_glob: str,
    val_ratio: float,
    test_ratio: float,
    output_dir: Path,
    default_prompt: str,
    max_workers: Optional[int],
    seed: Union[int, float, str, bytes, bytearray, None],
):
    assert_split_ratio(val_ratio=val_ratio, test_ratio=test_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(root_dir.glob(image_glob))

    assert len(image_paths) > 0, "No images found"

    mask_paths = list(root_dir.glob(masks_glob))

    assert len(mask_paths) > 0, "No masks found"

    assert len(image_paths) == len(
        mask_paths
    ), "Number of images and masks do not match"

    # Sort the image and mask paths for alignment
    image_paths.sort()
    mask_paths.sort()

    normal_image_mask_paths = []
    benign_image_mask_paths = []
    malignant_image_mask_paths = []

    image_mask_paths = tuple(zip(image_paths, mask_paths))

    for image_mask_path in image_mask_paths:
        image_path, _ = image_mask_path

        us_type = image_path.stem.rsplit("_", 1)[-1]

        if us_type == "normal":
            normal_image_mask_paths.append(image_mask_path)
        elif us_type == "benign":
            benign_image_mask_paths.append(image_mask_path)
        else:
            malignant_image_mask_paths.append(image_mask_path)

    # Set the random seed
    random.seed(seed)

    _, normal_val_data, normal_test_data = get_train_val_test(
        normal_image_mask_paths, val_ratio, test_ratio
    )

    _, benign_val_data, benign_test_data = get_train_val_test(
        benign_image_mask_paths, val_ratio, test_ratio
    )

    _, malignant_val_data, malignant_test_data = get_train_val_test(
        malignant_image_mask_paths, val_ratio, test_ratio
    )

    val_data = set(normal_val_data + benign_val_data + malignant_val_data)
    test_data = set(normal_test_data + benign_test_data + malignant_test_data)

    train_json = []
    val_json = []
    test_json = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures_to_image_mask_path = {}
        for i, image_mask_path in enumerate(image_mask_paths, 1):
            future = executor.submit(
                get_single_json, default_prompt, i, image_mask_path
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
                if image_mask_path in val_data:
                    val_json.append(op)
                elif image_mask_path in test_data:
                    test_json.append(op)
                else:  # for train data
                    train_json.append(op)

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

    parser = argparse.ArgumentParser(
        description="Generate the json files for BUS dataset"
    )

    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("/mnt/Enterprise/PUBLIC_DATASETS/us_images/br-usg"),
        help="The root directory of the dataset",
    )

    parser.add_argument(
        "--image-glob",
        type=str,
        default="images/*.png",
        help="The glob pattern for the images",
    )

    parser.add_argument(
        "--masks-glob",
        type=str,
        default="masks/*.png",
        help="The glob pattern for the masks",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="The ratio of the validation set",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="The ratio of the test set",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="The output directory for the json files",
        required=True,
    )

    parser.add_argument(
        "--default-prompt",
        type=str,
        default="",
        help="The default prompt for the images",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="The maximum number of workers for the process pool executor",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for the random number generator",
    )

    args = parser.parse_args()

    main(**vars(args))
