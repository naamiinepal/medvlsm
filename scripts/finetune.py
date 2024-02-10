# ######################################
# # FULL-TRAINING SEGMENTATION CONFIGS #
# ######################################

import os
from operator import itemgetter
from default_configs import *

# TODO: Change configs based on the requirements of the experiment
# For references, go the sibling python file "default_configs.py".

# CUSTOM CONFIGS BLOCK -- start:
# dataset_prompts = {
#     "kvasir_polyp": non_rad_prompts,
#     "bkai_polyp": non_rad_prompts, 
#     "clinicdb_polyp": non_rad_prompts,
#     "isic": non_rad_prompts,
#     "dfu": non_rad_prompts,
#     "camus": camus_prompts,
#     "busi": busi_prompts,
#     "chexlocalize": chexlocalze_prompts,
#     "pooled_polyp": non_rad_prompts,
#     "pooled_all": ["random"]
# }

models = [
    "clipseg_adapter",
    "cris",
    "biomed_clipseg",
    "biomed_clipseg_d"
]
precision=32
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
