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
    "san": {"batch_size": 8, "lr": 0.001}
}


prompts = ["random"]
dataset_prompts = {
    "kvasir_polyp": prompts,
    "bkai_polyp": prompts, 
    "clinicdb_polyp": prompts,
    "isic": prompts,
    "dfu": prompts,
    "camus": prompts,
    "busi": prompts,
    "chexlocalize": prompts,
}

models = [
    # "san"
    "clipseg",
    # "clipseg_shallow_adapter_v",
    # "clipseg_shallow_adapter_vl",
    # "clipseg_shallow_adapter_vlc",
    # "clipseg_dense_adapter_v",
    # "clipseg_dense_adapter_vl",
    # "clipseg_dense_adapter_vlc",
]

precision=32
devices=[0]
# freeze_encoder = True


# CUSTOM CONFIGS BLOCK -- end:


for model in models:
    # Model specific cfgs
    cfg = models_configs[model]
    batch_size, lr = itemgetter("batch_size", "lr")(cfg)

    for dataset, prompts in dataset_prompts.items():
        for p in prompts:
            command = f"python src/train.py \
                experiment={model}.yaml \
                experiment_name={model}_{dataset}_{p} \
                datamodule=img_txt_mask/{dataset}.yaml \
                datamodule.batch_size={batch_size} \
                model.optimizer.lr={lr} \
                trainer.accelerator={accelerator} \
                trainer.precision={precision} \
                trainer.devices={devices} \
                prompt_type={p} \
                logger=csv.yaml \
                tags='[{model}, {dataset}, finetune, {p}]' \
                output_masks_dir=output_masks/{model}/ft/{dataset}/{p}"

            if debugger:
                command = f"{command} debug=default"

            # Log command in terminal
            print(f"RUNNING COMMAND \n{command}")

            # Run the command
            if os.system(command=command) != 0:
                print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
                exit()
