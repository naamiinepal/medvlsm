#!/bin/bash

source /mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/.venv/bin/activate

for i in {0..6}; do
    ROOT_DIR=CRIS.pytorch/exp/busi_80_10_10_p${i}_$1/CRIS_R50
    python scripts/eval_metrics.py --seg_path $ROOT_DIR/pred --gt_path $ROOT_DIR/gt --csv_path $ROOT_DIR/metrics.csv
done
