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
    "clipseg": {"batch_size": 64, "lr": 0.002},
    "clipseg_v_adapter": {"batch_size": 64, "lr": 0.002},
    "clipseg_vl_adapter": {"batch_size": 64, "lr": 0.002},
    "clipseg_vlc_adapter": {"batch_size": 64, "lr": 0.002},
    "biomed_clipseg": {"batch_size": 128, "lr": 0.002},
    "biomed_clipseg_d": {"batch_size": 128, "lr": 0.002},
    "cris": {"batch_size": 32, "lr": 0.00002},
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
    # "clipseg",
    "clipseg_v_adapter",
    "clipseg_vl_adapter",
    "clipseg_vlc_adapter",
    # "cris",
    # "biomed_clipseg",
    # "biomed_clipseg_d"
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
                experiment_name={model}_ft_{dataset}_{p} \
                datamodule=img_txt_mask/{dataset}.yaml \
                datamodule.batch_size={batch_size} \
                model.optimizer.lr={lr} \
                trainer.accelerator={accelerator} \
                trainer.precision={precision} \
                trainer.devices={devices} \
                prompt_type={p} \
                logger=wandb.yaml \
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
