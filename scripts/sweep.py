# ######################################
# # FULL-TRAINING SEGMENTATION CONFIGS #
# ######################################

import os
from operator import itemgetter
import random

# TODO: Change configs based on your compatibility
accelerator = "gpu"
devices = [0]
precision = "16-mixed"
debugger = False
models = [
    "clipseg",
    # "cris",
    # "biomedclipseg",
    # "biomedclipseg_d"
]

non_rad_prompts = [f"p{i}" for i in range(10)]
chexlocalze_prompts = [f"p{i}" for i in range(7)]
camus_prompts = [f"p{i}" for i in range(8)]
busi_prompts = [f"p{i}" for i in range(7)]

dataset_prompts = {
    "kvasir_polyp": non_rad_prompts,
    "bkai_polyp": non_rad_prompts,
    "clinicdb_polyp": non_rad_prompts,
    "isic": non_rad_prompts,
    "dfu": non_rad_prompts,
    "camus": camus_prompts,
    "busi": busi_prompts,
    "chexlocalize": chexlocalze_prompts,
}

for model in models:
    for dataset, prompts in dataset_prompts.items():
        p = random.choice(prompts)
            # -m hparams_search={model}_optuna.yaml \
        command = f"python src/train.py \
            experiment={model}.yaml \
            experiment_name={model}_sweep \
            datamodule=img_txt_mask/{dataset}.yaml \
            prompt_type={p} \
            tags='[{model}, {dataset}, sweep, {p}]' \
            trainer.accelerator={accelerator} \
            trainer.devices={devices} \
            trainer.precision={precision}"

        if debugger:
            command = f"{command} debug=default"

        # Log command in terminal
        print(command)

        # Run the command
        if os.system(command=command) != 0:
            print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
            exit()
