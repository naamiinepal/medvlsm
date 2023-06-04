#! /bin/bash

for i in {1..3}; do
    python scripts/convert_sitk_to_image.py \
        --image-root /mnt/Enterprise/PUBLIC_DATASETS/camus_database/training \
        --glob-pattern 'patient*_?CH_E?_gt.mhd' \
        --out-dir /mnt/Enterprise/PUBLIC_DATASETS/camus_database_png/training/mask_$i \
        --out-ext .png --mask-index $i --resample-size 352
done
