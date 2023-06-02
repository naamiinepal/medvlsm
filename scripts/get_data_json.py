import json
import os
import random
from typing import Union

from tqdm import tqdm

from features_from_img import mask_to_overall_bbox

StrPath = Union[str, os.PathLike]

random.seed(444)


def get_json_data(
    root_dir: StrPath,
    img_dir_name: StrPath,
    mask_dir_name: StrPath,
    op_dir: StrPath,
    val_per: float,
    test_per: float,
    default_prompt: str,
):
    """Genrates Annotations of segmentation datset in the format requitred from training pytorch.CRIS
    Assumes that we have a root_dir, inside which there are images and masks directories and the images are named sequentially from 0 inside these dirctories,
    Also, this is for the assumption that each image has a single annotation decribing all the objets.

    Args:
        root_dir (string): path of the root dirctory, relative to which directories of images and masks are located
        mask_dir_name (string): name or path or dircetory with segmentation masks, relative to the root_dir
        op_dir (string): path of the json file of annotations to be output
        val_per(float): valid split percentage
        test_per(float): test split percentage

    Returns:
        string: success message
    """

    assert (val_per + test_per) <= 1.0, "val_per + test_per should be less than 1.0"

    total_data = os.listdir(os.path.join(root_dir, mask_dir_name))
    # total_data = [x for x in total_data if int(x[:-4]) >= 900]

    val_data = set(random.sample(total_data, int(val_per * len(total_data))))
    rem_data = {x for x in total_data if x not in val_data}
    test_data = set(random.sample(tuple(rem_data), int(test_per * len(total_data))))
    train_data = {x for x in rem_data if x not in test_data}

    train_json = []
    val_json = []
    test_json = []

    img_ext = os.listdir(os.path.join(root_dir, img_dir_name))[0].rsplit(".", 1)[-1]

    for i, mask in enumerate(tqdm(total_data)):
        mask_path = os.path.join(root_dir, mask_dir_name, mask)
        bbox = mask_to_overall_bbox(mask_path)
        seg_id = i

        img_name = mask.split(".")[0] + "." + img_ext
        sent = [{"idx": 0, "sent_id": i, "sent": default_prompt}]
        op = {
            "bbox": bbox,
            "cat": 0,
            "segment_id": seg_id,
            "img_name": img_name,
            "mask_name": mask,
            "sentences": sent,
            "sentences_num": 1,
        }

        if mask in train_data:
            train_json.append(op)
        elif mask in val_data:
            val_json.append(op)
        elif mask in test_data:
            test_json.append(op)

    print(len(train_json), len(val_json), len(test_json))

    os.makedirs(op_dir, exist_ok=True)

    with open(os.path.join(op_dir, "train.json"), "w") as of:
        json.dump(train_json, of)
    with open(os.path.join(op_dir, "val.json"), "w") as of:
        json.dump(val_json, of)
    with open(os.path.join(op_dir, "testA.json"), "w") as of:
        json.dump(test_json, of)
    with open(os.path.join(op_dir, "testB.json"), "w") as of:
        json.dump(test_json, of)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_data_dir", type=str, required=True)
    parser.add_argument("--valid_per", type=float, required=True)
    parser.add_argument("--test_per", type=float, required=True)
    # parser.add_argument("--cls_name", type=str, required=True)
    parser.add_argument("--default_prompt", type=str, required=False)
    args = parser.parse_args()

    get_json_data(
        args.root_data_dir,
        args.image_dir,
        args.mask_dir,
        args.output_data_dir,
        args.valid_per,
        args.test_per,
        # args.cls_name,
        args.default_prompt,
    )

    print("Annotations Created")
