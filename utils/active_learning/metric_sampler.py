import json
import os
from argparse import ArgumentParser

import pandas as pd


def main(sampling_frac, ds_root, op_root, ds_name):
    with open(f"{ds_root}/{ds_name}/anns/train.json") as train_data:
        total_data = json.load(train_data)
    
    sampling_frac = float(sampling_frac)
    
    print(f"Current Sampling Fraction: {sampling_frac}")
    sampling_fracs = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    frac_num_maps = {}
    
    for frac in sampling_fracs:
        frac_num_maps[f"{frac}"]=int(len(total_data)*frac)
    
    curr_idx = sampling_fracs.index(sampling_frac)
    prev_frac = sampling_fracs[curr_idx-1]
    
    num_samples = frac_num_maps[f"{sampling_frac}"] - frac_num_maps[f"{prev_frac}"]
    
    print(f"Selecting Samples: {num_samples}")

    models =["cris", "clipseg"]

    for model in models:
    
        metric_csv= f"{op_root}/{model}/al_ms/al_ms_{str(prev_frac)}/{ds_name}/train/consistency.csv"

        print(f"Using {metric_csv}")

        csv_df = pd.read_csv(metric_csv)
        csv_df.sort_values(by=['multi_iou'], ascending=True, inplace=True)
        filenames = list(csv_df.iloc[0:int(num_samples)]['filename'].reset_index(drop=True))

        prev_frac_filenames_path = f"{ds_root}/{ds_name}/anns_multiiou_sampled_{model}/train_{prev_frac}_filenames.json"


        if os.path.exists(prev_frac_filenames_path):
            print(f"Updating to previous: {prev_frac_filenames_path}")
            with open(prev_frac_filenames_path) as prev_data:
                prev_filenames = json.load(prev_data)
                filenames = filenames + prev_filenames

        frac_data = []
        rest_data = []

        for data in total_data:
            if data['mask_name'] in filenames:
                frac_data.append(data)
            else:
                rest_data.append(data)

        with open(f"{ds_root}/{ds_name}/anns_multiiou_sampled_{model}/train_{sampling_frac}.json", "w") as op:
            op.write(json.dumps(frac_data))

        with open(f"{ds_root}/{ds_name}/anns_multiiou_sampled_{model}/train_{sampling_frac}_filenames.json", "w") as op:
            op.write(json.dumps(filenames))

        with open(f"{ds_root}/{ds_name}/anns_multiiou_sampled_{model}/test_{sampling_frac}.json", "w") as op:
            op.write(json.dumps(rest_data))

        print("Sucessfully Sampled")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--sampling_frac",
        type=float,
        default=None,
        help="fraction of total training data to be sampled",
    )
    parser.add_argument(
        "--ds_root",
        type=str,
        default=None,
        help="root directory where the datasets are stored",
    )
    parser.add_argument(
        "--op_root",
        type=str,
        default=None,
        help="root directory where outputs are stored",
    )
    parser.add_argument(
        "--ds_name",
        type=str,
        default=None,
        help="name of the dataset",
    )
    args = parser.parse_args()

    main(**vars(args))
