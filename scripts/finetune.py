# ######################################
# # FULL-TRAINING SEGMENTATION CONFIGS #
# ######################################

import os
from operator import itemgetter
from default_configs import *

# TODO: Change configs based on the requirements of the experiment
# For references, go the sibling python file "default_configs.py".

# CUSTOM CONFIGS BLOCK -- start:
models_configs = {
    "clipseg": {"batch_size": 32, "lr": 0.001},
    "clipseg_shallow_adapter_v": {"batch_size": 32, "lr": 0.001},
    "clipseg_shallow_adapter_vl": {"batch_size": 32, "lr": 0.001},
    "clipseg_shallow_adapter_vlc": {"batch_size": 32, "lr": 0.001},
    "clipseg_dense_adapter_v": {"batch_size": 32, "lr": 0.001},
    "clipseg_dense_adapter_vl": {"batch_size": 32, "lr": 0.001},
    "clipseg_dense_adapter_vlc": {"batch_size": 32, "lr": 0.001},
    "san": {"batch_size": 32, "lr": 0.0003},
    "cris": {"batch_size": 32, "lr": 0.001},
}


prompts = ["random"]
datasets = [
    "kvasir_polyp",
    "bkai_polyp", 
    "clinicdb_polyp",
    "isic",
    "dfu",
    "camus",
    "busi",
    "chexlocalize",
]

models = [
    "cris",
    "clipseg_dense_adapter_v",
    "clipseg_dense_adapter_vl",
    "clipseg_dense_adapter_vlc",
    "clipseg_shallow_adapter_v",
    "clipseg_shallow_adapter_vl",
    "clipseg_shallow_adapter_vlc",
    "san",
    "clipseg",
]

# prompt_type = "random"
# precision="16-mixed"
# devices=[0]

# CUSTOM CONFIGS BLOCK -- end:


# Loops for running the finetune experiments
for model in models:
    for dataset in datasets:
        # Model specific cfgs
        cfg = models_configs[model]
        batch_size, lr = itemgetter("batch_size", "lr")(cfg)
        for seed in seeds:
            # Define bash command here for individual experiments
            command = f"python src/train.py \
                experiment={model}.yaml \
                experiment_name={model}_{dataset}_seed_{seed} \
                datamodule=img_txt_mask/{dataset}.yaml \
                datamodule.batch_size={batch_size} \
                model.optimizer.lr={lr} \
                trainer.accelerator={accelerator} \
                trainer.precision={precision} \
                trainer.devices={devices} \
                prompt_type={prompt_type} \
                seed={seed} \
                logger=wandb.yaml \
                tags='[{model}, {dataset}, {prompt_type}, seed_{seed}]' \
                output_masks_dir=output_masks/{model}/{dataset}/{prompt_type}"

            if debugger:
                command = f"{command} debug=default"

            # Log command in terminal
            print(f"RUNNING COMMAND \n{command}")

            # Run the defined command
            if os.system(command=command) != 0:
                # Block to run after encountering error
                print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
                exit()
