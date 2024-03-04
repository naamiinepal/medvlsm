#!/bin/bash

python3 utils/get_data_json.py --root_data_dir /mnt/Enterprise2/kanchan/medvlsm/datasets/bkai_polyp --dataset_name bkai_polyp --image_dir images --mask_dir masks --output_data_dir /mnt/Enterprise2/kanchan/medvlsm/datasets/bkai_polyp --valid_per 0.1 --test_per 0.1 