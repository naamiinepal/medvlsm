import json
import os
import random

from features_from_img import get_mask_decription, mask_to_overall_bbox


def get_json_data(root_dir, mask_dir_name, img_dir_name, op_dir, val_per, test_per):
    """Genrates Annotations of segmentation datset in the format requitred from training pytorch.CRIS
    Assumes that we have a root_dir, inside which there are images and masks directories and the images are named sequentially from 0 inside these dirctories,
    Also, this is for the assumption that each image has a single annotation decribing all the objets.

    Args:
        root_dir (string): path of the root dirctory, relative to which directories of images and masks are located
        mask_dir_name (string): name or path or dircetory with segenattion masks, rekative to the root_diir
        op_dir (string): path of the json file of annotations to be output
        val_per(float): valid split percentage
        test_per(float): test split percentage

    Returns:
        string: success message
    """
    random.seed(444)
    total_data = os.listdir(os.path.join(root_dir, mask_dir_name))
    print(val_per)

    val_data = random.sample(total_data, int(val_per * len(total_data)))
    rem_data = [x for x in total_data if x not in val_data]
    test_data = random.sample(rem_data, int(test_per * len(total_data)))
    train_data = [x for x in rem_data if x not in test_data]

    train_json = []
    val_json = []
    test_json = []

    img_ext = os.listdir(os.path.join(root_dir, img_dir_name))[0].split(".")[-1]

    for i, mask in enumerate(total_data):
        mask_path = os.path.join(root_dir, mask_dir_name, mask)
        bbox = mask_to_overall_bbox(mask_path)
        stat, prompt = get_mask_decription(mask_path)
        seg_id = int(mask_path.split("/")[-1].split(".")[0])
        img_name = mask.split(".")[0] + "." + img_ext
        sent = [{"idx": 0, "sent_id": i, "sent": prompt}]
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
    # with open(os.path.join(op_dir, "anns/isic/train.json"), "w") as of:
    #     of.write(json.dumps(train_json))
    # with open(os.path.join(op_dir, "anns/isic/val.json"), "w") as of:
    #     of.write(json.dumps(val_json))
    with open(os.path.join(op_dir, "anns/isic/testA.json"), "w") as of:
        of.write(json.dumps(test_json))
    with open(os.path.join(op_dir, "anns/isic/testB.json"), "w") as of:
        of.write(json.dumps(test_json))


get_json_data(
    "/mnt/Enterprise/PUBLIC_DATASETS/skin_datasets/isic/test",
    "masks_cf",
    "images_cf",
    "/mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/datasets",
    0.0,
    1.0,
)
