"""
Script to evaluate the segmentation metrics.
The script takes in the path to the segmentation and ground truth images and
computes the following metrics:
1. Surface Dice
2. Hausdorff Distance
3. IoU
4. Dice

NOTE: The script assumes that the segmentation and ground truth images 
have the same name. The script also assumes that the images are binary
images with pixel values 0 and 255. The script thresholds the images to 0 and 1
and computes the metrics. The script also assumes that the images are of size
[H, W] and not [H, W, C], and are of type uint8.

The script saves the metrics in a csv file.

Usage:
    python eval_metrics.py \
        --seg_path <path to segmentation images> \
        --gt_path <path to ground truth images> \
        --csv_path <path to save csv file>
"""


import os
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
import pandas as pd
import torch
from monai.metrics import (
    compute_dice,
    compute_hausdorff_distance,
    compute_iou,
    compute_surface_dice,
)
from tqdm import tqdm
import concurrent.futures


def compute_metrics(gt_img_path: str, pred_img_path: str):
    gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)

    # make sure the images are of same size
    assert (
        gt_img.shape == pred_img.shape
    ), f"Images {gt_img_path} and {pred_img_path} are of different sizes"

    # threshold the images
    gt_img[gt_img > 0] = 1
    pred_img[pred_img > 0] = 1

    # change images to batch-first tensor [B,C,H,W]
    gt_img = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0)
    pred_img = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0)

    # compute the metrics
    surface_dice = compute_surface_dice(pred_img, gt_img, class_thresholds=[0.5])
    hausdorff_distance = compute_hausdorff_distance(pred_img, gt_img)
    iou = compute_iou(pred_img, gt_img)
    dice = compute_dice(pred_img, gt_img)

    return surface_dice.item(), hausdorff_distance.item(), iou.item(), dice.item()


def main(args: Namespace):
    surface_dice_list = []
    hausdorff_distance_list = []
    iou_list = []
    dice_list = []

    np.set_printoptions(precision=4)

    filenames = tuple(x for x in os.listdir(args.seg_path) if x.endswith(".png"))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {}
        for filename in filenames:
            gt_img_path = os.path.join(args.gt_path, filename)
            pred_img_path = os.path.join(args.seg_path, filename)

            futures[
                executor.submit(compute_metrics, gt_img_path, pred_img_path)
            ] = filename

        with tqdm(total=len(futures), desc="Evaluating metrics") as pbar:
            for future in tqdm(concurrent.futures.as_completed(futures)):
                filename = futures[future]
                try:
                    surface_dice, hausdorff_distance, iou, dice = future.result()
                except Exception as exc:
                    print(f"{filename} generated an exception: {exc}")
                else:
                    surface_dice_list.append(surface_dice)
                    hausdorff_distance_list.append(hausdorff_distance)
                    iou_list.append(iou)
                    dice_list.append(dice)

                pbar.set_postfix(
                    {
                        "Mean Dice": np.mean(dice_list),
                        "Mean IoU": np.mean(iou_list),
                    }
                )

    # with tqdm(filenames, desc="Evaluating metrics") as pbar:
    #     for filename in pbar:
    #         gt_img_path = os.path.join(args.gt_path, filename)
    #         pred_img_path = os.path.join(args.seg_path, filename)

    #         surface_dice, hausdorff_distance, iou, dice = compute_metrics(
    #             gt_img_path, pred_img_path
    #         )

    #         surface_dice_list.append(surface_dice)
    #         hausdorff_distance_list.append(hausdorff_distance)
    #         iou_list.append(iou)
    #         dice_list.append(dice)

    #         pbar.set_postfix(
    #             {
    #                 "file": filename,
    #                 "Mean Dice": np.mean(dice_list),
    #                 "Mean IoU": np.mean(iou_list),
    #             }
    #         )

    df = pd.DataFrame(
        {
            "filename": filenames,
            "surface_dice": surface_dice_list,
            "hausdorff_distance": hausdorff_distance_list,
            "iou": iou_list,
            "dice": dice_list,
        }
    )

    print("Mean surface dice: ", np.mean(surface_dice_list))
    print("Std surface dice: ", np.std(surface_dice_list))

    print("Mean hausdorff distance: ", np.mean(hausdorff_distance_list))
    print("Std hausdorff distance: ", np.std(hausdorff_distance_list))

    print("Mean iou: ", np.mean(iou_list))
    print("Std iou: ", np.std(iou_list))

    print("Mean dice: ", np.mean(dice_list))
    print("Std dice: ", np.std(dice_list))

    df.to_csv(args.csv_path, index=False, float_format="%.4f")
    print(f"Saved metrics to {args.csv_path}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--seg_path",
        type=str,
        default="/mnt/Enterprise/safal/VLM-SEG-2023/testdata/pred",
        help="path to segmentation files",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="/mnt/Enterprise/safal/VLM-SEG-2023/testdata/gt",
        help="path to ground truth files",
    )
    parser.add_argument(
        "--csv_path", type=str, default="metrics.csv", help="path to save csv file"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="maximum number of workers to use for multiprocessing",
    )

    args = parser.parse_args()
    main(args)
