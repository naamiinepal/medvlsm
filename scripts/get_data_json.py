import argparse
import ast
import json
import os
import random

import pandas as pd
from features_from_img import mask_to_overall_bbox


def get_json_data(
    root_dir,
    dataset,
    img_dir_name,
    mask_dir_name,
    op_dir,
    val_per,
    test_per,
    #anns_name,
    prompt_csv=None,
    default_prompt="",
):
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
    # total_data = [x for x in total_data if int(x[:-4]) >= 900]

    val_data = random.sample(total_data, int(float(val_per) * len(total_data)))
    rem_data = [x for x in total_data if x not in val_data]
    test_data = random.sample(rem_data, int(float(test_per) * len(total_data)))
    train_data = [x for x in rem_data if x not in test_data]

    train_json = []
    val_json = []
    test_json = []

    img_ext = os.listdir(os.path.join(root_dir, img_dir_name))[0].split(".")[-1]

    #for dataset in ['Kvaisir-SEG', 'clinicdb-polyp', 'bkai-polyp', 'cvc-300-polyp', 'cvc-colondb-polyp', 'etis-polyp']:
    with open(f"/mnt/Enterprise/miccai_2023_CRIS/others/prompts_csv/{dataset}.csv") as prompt_csv:

        prompt_df = pd.read_csv(prompt_csv)
        prompt_df["p0"] = prompt_df["p0"].fillna("")

        # prompt_df["p7"] = [n.strip() for n in ast.literal_eval(prompt_df["p7"])]

        prompt_df["p7"] = prompt_df["p7"].apply(
            lambda x: [n.strip() for n in ast.literal_eval(x)]
        )
        prompt_df["p8"] = prompt_df["p8"].apply(
            lambda x: [n.strip() for n in ast.literal_eval(x)]
        )
        prompt_df["p9"] = prompt_df["p9"].apply(
            lambda x: [n.strip() for n in ast.literal_eval(x)]
        )

        for i, mask in enumerate(total_data):
            mask_path = os.path.join(root_dir, mask_dir_name, mask)
            bbox = mask_to_overall_bbox(mask_path)
            prompt = default_prompt
            seg_id = int(mask_path.split("/")[-1].split(".")[0])

            img_name = mask.split(".")[0] + "." + img_ext
            sent = [{"idx": 0, "sent_id": i, "sent": prompt}]

            mask_df = prompt_df[prompt_df["masks"] == mask]
            mask_df = mask_df.drop(columns=["image", "masks"])
            prompts = mask_df.iloc[0].to_dict()
            op = {
                "bbox": bbox,
                "cat": 0,
                "segment_id": seg_id,
                "img_name": img_name,
                "mask_name": mask,
                "sentences": sent,
                "prompts": prompts,
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
            of.write(json.dumps(train_json))
        with open(os.path.join(op_dir, "val.json"), "w") as of:
            of.write(json.dumps(val_json))
        with open(os.path.join(op_dir, "testA.json"), "w") as of:
            of.write(json.dumps(test_json))
        with open(os.path.join(op_dir, "testB.json"), "w") as of:
            of.write(json.dumps(test_json))
        print(f'Json making completed for {dataset}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_data_dir", type=str, required=True)
    parser.add_argument("--valid_per", type=str, required=True)
    parser.add_argument("--test_per", type=str, required=True)
    #parser.add_argument("--anns_name", type=str, required=True)
    # parser.add_argument("--cls_name", type=str, required=True)
    parser.add_argument("--default_prompt", type=str, required=False)
    args = parser.parse_args()

    print(args)
    get_json_data(
        args.root_data_dir,
        args.dataset_name,
        args.image_dir,
        args.mask_dir,
        args.output_data_dir,
        args.valid_per,
        args.test_per,
        # args.cls_name,
        #args.anns_name,
        args.default_prompt,
    )

    print("Annotations Created")
