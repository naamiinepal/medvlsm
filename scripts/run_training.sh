#! /bin/bash

chgrp -R miccai_2023 /mnt/Enterprise/miccai_2023_CRIS/

source /mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/.venv/bin/activate

python -u /mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/train.py --config /mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/config/kvasir_polyp/cris_r50_v5.yaml
