import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import pandas as pd
from features_from_img import get_mask_description
from PIL import Image


def get_prompts_csv(
    dataset_root,
    out_dir,
    img_dir,
    msk_dir,
    class_name,
    ds_name,
    general_description_list,
):
    image_dir = os.path.join(dataset_root, img_dir, "*")
    mask_dir = os.path.join(dataset_root, msk_dir, "*")

    size = location = number = ""

    img_name_list = []
    mask_name_list = []
    p0_list = []
    p1_list = []
    p2_list = []
    p3_list = []
    p4_list = []
    p5_list = []
    p6_list = []
    p7_list = []
    p8_list = []
    p9_list = []

    images_fn = sorted(glob(image_dir))
    masks_fn = sorted(glob(mask_dir))

    for idx, image_fn in enumerate(images_fn):
        print(images_fn[idx], image_fn)
        img_name_list.append(images_fn[idx].split("/")[-1])
        mask_name_list.append(masks_fn[idx].split("/")[-1])
        
        size, location, number = get_mask_description(masks_fn[idx])

        p0_list.append(f"")
        p1_list.append(f"{class_name}")
        p2_list.append(f"{size} {class_name}")
        p3_list.append(f"{number} {class_name}")
        p4_list.append(
            f"{class_name}, located in {location} of the image"
        )
        p5_list.append(f"{number} {size} {class_name}")
        
        p6_list.append(f"{size} {class_name}, located in {location} of the image")

        p7_list.append(
                        f"{number} {class_name}, located in {location} of the image"
        )
        p8_list.append(f"{number} {size} {class_name}, located in {location} of the image")

        p9_list.append(
            [
                f"{number} {size} {general_description} located in {location} of the image"
                for general_description in general_description_list
            ]
        )

        csv_df = pd.DataFrame()
        csv_df["image"] = img_name_list
        csv_df["masks"] = mask_name_list
        csv_df["p0"] = p0_list
        csv_df["p1"] = p1_list
        csv_df["p2"] = p2_list
        csv_df["p3"] = p3_list
        csv_df["p4"] = p4_list
        csv_df["p5"] = p5_list
        csv_df["p6"] = p6_list
        csv_df["p7"] = p7_list
        csv_df["p8"] = p8_list
        csv_df["p9"] = p9_list

        csv_df.to_csv(f"{out_dir}/{ds_name}.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="dataset root directory, relative to which image and masks are present",
    )
    parser.add_argument(
        "--csv_out_dir", type=str, required=True, help="directory to dump output csvs"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="image directory relative to datset_root",
    )
    parser.add_argument(
        "--msk_dir",
        type=str,
        required=True,
        help="mask directory relative to datset_root",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        help="class name keyword attribute's value for prompt",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="name of the dataset"
    )
    parser.add_argument(
        "--descriptions_json_file",
        type=str,
        required=True,
        help="json file with the list of general descriptions of the class",
    )

    args = parser.parse_args()

    desciptions_list = []
    with open(args.descriptions_json_file, "r") as f:
        desciptions_list = json.load(f)

    get_prompts_csv(
        args.dataset_root,
        args.csv_out_dir,
        args.img_dir,
        args.msk_dir,
        args.class_name,
        args.dataset_name,
        desciptions_list,
    )
