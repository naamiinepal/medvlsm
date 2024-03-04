import concurrent.futures
import math
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou
from tqdm import tqdm


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
    gt_img = torch.from_numpy(gt_img)[None, None, ...]
    pred_img = torch.from_numpy(pred_img)[None, None, ...]

    # compute the metrics
    # surface_dice = compute_surface_dice(pred_img, gt_img, class_thresholds=[0.5]) * 100
    iou = compute_iou(pred_img, gt_img, ignore_empty=False) * 100
    dice = compute_dice(pred_img, gt_img, ignore_empty=False) * 100
    
    return {
        "iou": iou.item(),
        "dice": dice.item(),
    }


def main(
    seg_path: Path,
    gt_path: Path,
    csv_path: Union[str, Path],
    max_workers: Optional[int],
):
    print(seg_path, gt_path, csv_path)
    np.set_printoptions(precision=5)
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count//(max_workers or cpu_count))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for filename in seg_path.glob("*.png"):
            gt_img_path = str(gt_path / filename.name)
            pred_img_path = str(seg_path / filename.name)

            futures[
                executor.submit(compute_metrics, gt_img_path, pred_img_path)
            ] = filename

        aggregator = defaultdict(list)

        with tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating metrics",
        ) as pbar:
            for future in pbar:
                filename = futures[future]
                try:
                    results = future.result()
                except Exception as exc:
                    print(f"{filename} generated an exception: {exc}")
                else:
                    aggregator["filename"].append(filename)
                    for key, value in results.items():
                        aggregator[key].append(value)

                pbar.set_postfix(
                    {
                        "Mean Dice": np.nanmean(aggregator["dice"]),
                    }
                )

    df = pd.DataFrame(aggregator)

    # print mean and std for each metric
    for key in df.columns:
        if key != "filename":
            print_mean_std(df, key)

    # sort the dataframe by filename to make output consistent
    df.sort_values(by="filename", inplace=True)

    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved metrics to {csv_path}")


def print_mean_std(df: pd.DataFrame, column_name: str):
    column = df[column_name]
    print(
        column_name.replace("_", " ").title(),
        "$",
        round(column.mean(), 2),
        "\smallStd{",
        round(column.std(), 2),
        "}$",
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--seg_path",
        type=Path,
        help="path to segmentation files",
    )
    parser.add_argument(
        "--gt_path",
        type=Path,
        help="path to ground truth files",
    )
    parser.add_argument(
        "--csv_path", type=str, default="metrics.csv", help="path to save csv file"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="maximum number of workers to use for multiprocessing",
    )

    args = parser.parse_args()

    main(**vars(args))
