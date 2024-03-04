#!/bin/bash

# Train the models with sdm_camus dataset
# Model configs

import os
from argparse import ArgumentParser
from operator import itemgetter
from typing import Literal

# TODO: Change configs based on your compatibility

# WARNING !! If want to update the configs, do not changes the "BASELINE CONFIG";
# there is a separete "CUSTOM CONFIG" block at the immediate end of the baseline block

#  BASELINE CONFIG -- Do not change this

def main(train_frac):
    train_frac = float(train_frac)
    accelerator = "gpu"
    devices = [0]
    precision = "16-mixed"
    debugger = False
    models = ["clipseg", "cris"]

    models_configs = {

    "clipseg": {"batch_size": 128, "lr": 0.002},
    "cris": {"batch_size": 32, "lr": 0.00002},
    }
    non_rad_prompts = [f"p{i}" for i in range(2, 10)]


    dataset_prompts = {
        "kvasir_polyp": non_rad_prompts,
    }

    task_name : Literal["pred", "eval", "train" ] = "eval" 
    # BASELINE CONFIGS -- End here


    # CUSTOM CONFIG -- start:
    # devices=[1]
    dataset_prompts = {
        "kvasir_polyp": non_rad_prompts,
    }



    task_name = "pred"
    # CUSTOM CONFIG -- end:

    if train_frac == 0.0:
        use_ckpt = "false"
    else:
        use_ckpt = "true"

    for model in models:
        # Model specific cfgs
        cfg = models_configs[model]
        batch_size, lr = itemgetter("batch_size", "lr")(cfg)

        for dataset, prompts in dataset_prompts.items():
            best_ckpt = f"logs/train/runs/{model}_al_ms_ft_{train_frac}_{dataset}_random/checkpoints/best.ckpt"
            for p in prompts:
                command = f"python src/eval.py \
                    experiment={model}.yaml \
                    experiment_name={model}_al_ms_infer_{train_frac}_{dataset}_{p} \
                    datamodule=img_txt_mask_al/{dataset}_samp.yaml \
                    prompt_type={p} \
                    train_frac={train_frac}\
                    model_name={model}\
                    datamodule.batch_size={batch_size} \
                    trainer.accelerator={accelerator} \
                    trainer.devices={devices} \
                    use_ckpt={use_ckpt} \
                    logger=csv \
                    output_masks_dir=output_masks/{model}/al_ms/al_ms_{train_frac}/{dataset}/train/{p} \
                    task_name={task_name}"
                if train_frac != 0:
                    command = command + f" ckpt_path={best_ckpt}"
                # Log command in terminal
                print(command)


                # Run the command
                if os.system(command=command) != 0:
                    print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
                    exit()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train_frac",
        type=float,
        default=None,
        help="fraction of training data used in the model to be used for inference",
    )

    args = parser.parse_args()

    main(**vars(args))
