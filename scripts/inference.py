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
models = ["clipseg", "cris", "biomed_clipseg", "biomed_clipseg_d"]

models_configs = {
    "clipseg": {"batch_size": 128, "lr": 0.002},
    "cris": {"batch_size": 32, "lr": 0.00002},
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
    "pooled_all": ["random"],
}

task_name: Literal["pred", "eval", "train"] = "eval"
# BASELINE CONFIGS -- End here


# CUSTOM CONFIG -- start:
devices = [0]


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
}

models = [
    # "clipseg",
    # "cris",
    "biomed_clipseg",
    "biomed_clipseg_d"
]

task_name = "eval"
ft_endo_datasets = {
    "kvasir_polyp": non_rad_prompts,
    "bkai_polyp": non_rad_prompts,
    "clinicdb_polyp": non_rad_prompts,
}

test_endo_datasets = {
    "kvasir_polyp": non_rad_prompts,
    "bkai_polyp": non_rad_prompts,
    "clinicdb_polyp": non_rad_prompts,
    "cvc300_polyp": non_rad_prompts,
    "colondb_polyp": non_rad_prompts,
    "etis_polyp": non_rad_prompts,
}

models_configs = {
    "clipseg": {"batch_size": 128, "lr": 0.002},
    "biomed_clipseg": {"batch_size": 128, "lr": 0.002},
    "biomed_clipseg_d": {"batch_size": 128, "lr": 0.002},
    "cris": {"batch_size": 32, "lr": 0.00002},
}

# CUSTOM CONFIG -- end:


for model in models:
    # Model specific cfgs
    cfg = models_configs[model]
    batch_size, lr = itemgetter("batch_size", "lr")(cfg)

    for dataset, prompts in dataset_prompts.items():
        for p in prompts:
            experiment_name = f"{model}_ft_{dataset}_{p}"
            ckpt_path = f"logs/train/runs/{model}_ft_{dataset}_{p}/checkpoints/best.ckpt"
            command = f"python src/eval.py \
                experiment={model}.yaml \
                experiment_name={experiment_name} \
                datamodule=img_txt_mask/{dataset}.yaml \
                prompt_type={p} \
                datamodule.batch_size={batch_size} \
                trainer.accelerator={accelerator} \
                trainer.devices={devices} \
                use_ckpt=True \
                ckpt_path={ckpt_path} \
                logger=csv \
                output_masks_dir=output_masks/{model}/pooled_polyp/{dataset}/{p} \
                task_name={task_name}"
            # Log command in terminal
            print(command)

            # Run the command
            if os.system(command=command) != 0:
                print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
                exit()


# for model in models:
#     # Model specific cfgs
#     cfg = models_configs[model]
#     batch_size, lr = itemgetter("batch_size", "lr")(cfg)

#     for ft_ds, _ in ft_endo_datasets.items():
#         for test_ds, prompts in test_endo_datasets.items():
#             if ft_ds == test_ds:
#                 continue
#             for p in prompts:
#                 ckpt_path = f"logs/train/runs/{model}_ft_{ft_ds}_{p}/checkpoints/best.ckpt"
#                 experiment_name = f"{model}_ft_on_{ft_ds}_tested_on_{test_ds}_{p}"
#                 command = f"python src/eval.py \
#                     experiment={model}.yaml \
#                     experiment_name={experiment_name} \
#                     datamodule=img_txt_mask/{test_ds}.yaml \
#                     prompt_type={p} \
#                     datamodule.batch_size={batch_size} \
#                     trainer.accelerator={accelerator} \
#                     trainer.devices={devices} \
#                     use_ckpt=True \
#                     ckpt_path={ckpt_path} \
#                     logger=csv \
#                     output_masks_dir=output_masks/{model}/X_datasets/ft_on_{ft_ds}/tested_on_{test_ds}/{p} \
#                     task_name={task_name}"
#                 # Log command in terminal
#                 print(command)


#                 # Run the command
#                 if os.system(command=command) != 0:
#                     print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
#                     exit()
