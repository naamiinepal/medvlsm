#!/bin/bash

fracs=(0.025)
root_dir= #abs dir of the medvlsm repo
ds=kvasir_polyp
for i in $(seq 1 8); do
    frac=${fracs[i]}
    prev_frac=${fracs[i-1]}
        python3 scripts/active_learning/infer_al.py \
                --train_frac=${prev_frac}
        python3 scripts/active_learning/eval_al.py \
                --seg_root_path=output_masks/clipseg/al_ms/al_ms_${prev_frac}/${ds}/train \
                --csv_path=output_masks/clipseg/al_ms/al_ms_${prev_frac}/${ds}/train/consistency.csv
        python3 scripts/active_learning/eval_al.py \
                --seg_root_path=output_masks/cris/al_ms/al_ms_${prev_frac}/${ds}/train \
                --csv_path=output_masks/cris/al_ms/al_ms_${prev_frac}/${ds}/train/consistency.csv
        python3 utils/active_learning/metric_sampler.py \
                --sampling_frac=${frac}
                --ds_root=${root_dir}/datasets
                --op_root=${root_dir}/output_masks
                --ds_name=${ds}
        python3 scripts/active_learning/finetune_al.py \
            --train_frac=${frac}
done
fracs=(0.025)

for frac in fracs; do
    frac=${fracs[i]}
    python3 scripts/active_learning/finetune_random.py \
        --train_frac=${frac}
done