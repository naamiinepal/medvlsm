#!/bin/bash

# Train the models with sdm_camus dataset
# Model configs

import os
import logging
from operator import itemgetter
from typing import Literal


# TODO: Change configs based on your compatibility

# WARNING !! If want to update the configs, do not changes the "BASELINE CONFIG";
# there is a separete "CUSTOM CONFIG" block at the immediate end of the baseline block

#  BASELINE CONFIG -- Do not change this
accelerator = "gpu"
devices = [0]
precision = "16-mixed"
debugger = False
models = ["clipseg", "cris", "biomedclipseg", "biomedclipseg_d", "zsref"]

models_configs = {
    "clipseg": {"batch_size": 128, "lr": 0.002},
    "cris": {"batch_size": 32, "lr": 0.00002},
    "zsref": {"batch_size": 1, "lr": 0.0002}
}
non_rad_prompts = [f"p{i}" for i in range(10)]
chexlocalze_prompts = [f"p{i}" for i in range(7)]
camus_prompts = [f"p{i}" for i in range(8)]
busi_prompts = [f"p{i}" for i in range(7)]

dataset_prompts = {
    "kvasir_polyp": non_rad_prompts,
    "bkai_polyp": non_rad_prompts,
    "clinicdb_polyp": non_rad_prompts,
    "cvc300_polyp": non_rad_prompts,
    "colondb_polyp": non_rad_prompts,
    "etis_polyp": non_rad_prompts,
    "isic": non_rad_prompts,
    "dfu": non_rad_prompts,
    "camus": camus_prompts,
    "busi": busi_prompts,
    "chexlocalize": chexlocalze_prompts,
    "pooled_all": ["random"]
}

task_name : Literal["pred", "eval", "train" ] = "eval" 
# BASELINE CONFIGS -- End here


# CUSTOM CONFIG -- start:
# devices=[1]
dataset_prompts = {
    "kvasir_polyp": non_rad_prompts,
    "bkai_polyp": non_rad_prompts,
    "clinicdb_polyp": non_rad_prompts,
    "cvc300_polyp": non_rad_prompts,
    "colondb_polyp": non_rad_prompts,
    "etis_polyp": non_rad_prompts,
    "isic": non_rad_prompts,
    "dfu": non_rad_prompts,
    "camus": camus_prompts,
    "busi": busi_prompts,
    "chexlocalize": chexlocalze_prompts,
    "pooled_all": ["random"]
}

models = [
    # "clipseg",
    "zsref",
    # "cris",
    # "biomedclipseg",
    # "biomedclipseg_d"
]

task_name = "pred"
# CUSTOM CONFIG -- end:


for model in models:
    # Model specific cfgs
    cfg = models_configs[model]
    batch_size, lr = itemgetter("batch_size", "lr")(cfg)

    for dataset, prompts in dataset_prompts.items():
        for p in prompts:
            experiment_name = f"{model}_zss_{dataset}_{p}"
            # ckpt_path = f"logs/train/runs/${experiment_name}/checkpoints/best.ckpt"
            command = f"python src/eval.py \
                experiment={model}.yaml \
                experiment_name={model}_zss_{dataset}_{p} \
                datamodule=img_txt_mask/{dataset}.yaml \
                prompt_type={p} \
                datamodule.batch_size={batch_size} \
                trainer.accelerator={accelerator} \
                trainer.devices={devices} \
                use_ckpt=false \
                logger=csv \
                output_masks_dir=output_masks/{model}/zss/{dataset}/{p} \
                task_name={task_name}"
            # Log command in terminal
            print(command)


            # Run the command
            if os.system(command=command) != 0:
                print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
                exit()
