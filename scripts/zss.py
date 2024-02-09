#!/bin/bash

# Train the models with sdm_camus dataset
# Model configs

import os
from operator import itemgetter
from default_configs import *

# TODO: Change configs based on the requirements of the experiment
# For references, go the sibling python file "default_configs.py".

# CUSTOM CONFIG -- start:
models = ["clipseg", "cris"]

# CUSTOM CONFIG -- end:


for model in models:
    # Model specific cfgs
    cfg = models_configs[model]
    batch_size, lr = itemgetter("batch_size", "lr")(cfg)

    for dataset, prompts in dataset_prompts.items():
        for p in prompts:
            experiment_name = f"{model}_zss_{dataset}_{p}"
            command = f"python src/eval.py \
                experiment={model}.yaml \
                experiment_name={experiment_name} \
                datamodule=img_txt_mask/{dataset}.yaml \
                prompt_type={p} \
                datamodule.batch_size={batch_size} \
                trainer.accelerator={accelerator} \
                trainer.devices={devices} \
                use_ckpt=False \
                output_masks_dir=output_masks/{model}/zss/{dataset}/{p} \
                task_name={task_name}"
            # Log command in terminal
            print(command)

            # Run the command
            if os.system(command=command) != 0:
                print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
                exit()
