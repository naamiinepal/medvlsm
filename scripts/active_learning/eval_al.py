import concurrent.futures
import glob
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
from monai.metrics import compute_hausdorff_distance  # ; compute_surface_dice,
from monai.metrics import compute_dice, compute_iou
from multi_iou import compute_multi_iou
from tqdm import tqdm


def compute_metrics(pred_img_paths: str):
    pred_imgs = np.stack((cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE) for pred_img_path in pred_img_paths), axis=0)

    # make sure the images are of same size
    #TODO

    # threshold the images
    pred_imgs[pred_imgs > 0] = 1

    # change images to  tensor [N,B,C,H,W]
    pred_imgs = torch.from_numpy(pred_imgs)[:, None, None, ...]

    # compute the metrics

    multi_iou = compute_multi_iou(pred_imgs)
 
    return {
        "multi_iou": multi_iou.item()
        }


def main(
    seg_root_path: Path,
    csv_path: Union[str, Path],
    max_workers: Optional[int],
):
    print(seg_root_path, csv_path)
    np.set_printoptions(precision=5)
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count//(max_workers or cpu_count))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for file_path in glob.glob(f"{os.path.join(seg_root_path, os.listdir(seg_root_path)[0])}/*.png"):
            filename = file_path.split('/')[-1]
            pred_img_paths = [f"{seg_root_path}/{p}/{filename}" for p in os.listdir(seg_root_path) if ".csv" not in p]
            futures[
                executor.submit(compute_metrics, pred_img_paths)
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
                        "Mean MultiIou": np.nanmean(aggregator["multiiou"]),
                        
                    }
                )

    df = pd.DataFrame(aggregator)

    # sort the dataframe by filename to make output consistent
    df.sort_values(by="filename", inplace=True)

    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved metrics to {csv_path}")




if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--seg_root_path",
        type=Path,
        help="path to segmentation files",
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
