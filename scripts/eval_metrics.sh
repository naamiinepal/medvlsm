#!/bin/bash

dataset='clinicdb_polyp_80_10_10'
version='_v0'

python eval_metrics.py --seg_path /mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/exp/$dataset$version/CRIS_R50/pred/ --gt_path /mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/exp/$dataset$version/CRIS_R50/gt/
