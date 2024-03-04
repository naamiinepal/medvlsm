# ######################################
# # FULL-TRAINING SEGMENTATION CONFIGS #
# ######################################

import os
from argparse import ArgumentParser
from operator import itemgetter

# TODO: Change configs based on your compatibility

# WARNING !! If want to update the configs, do not changes the "BASELINE CONFIG";
# there is a separete "CUSTOM CONFIG" block at the immediate end of the baseline block

#  BASELINE CONFIG -- Do not change this
def main(train_frac):
    accelerator = "gpu"
    devices = [0]
    precision = "16-mixed"
    debugger = False
    models = ["clipseg", "cris"]

    models_configs = {

    "clipseg": {"batch_size": 16, "lr": 0.002},
    "cris": {"batch_size": 16, "lr": 0.00002},
    }
    dataset_prompts = {
        "kvasir_polyp": ["random"]
    }
    # BASELINE CONFIGS -- End here


    # CUSTOM CONFIG -- start:


    # CUSTOM CONFIG -- end:

    train_frac =float(train_frac)
    for model in models:
        # Model specific cfgs
        cfg = models_configs[model]
        batch_size, lr = itemgetter("batch_size", "lr")(cfg)

        for dataset, prompts in dataset_prompts.items():
            for p in prompts:
                command = f"python src/train.py \
                    experiment={model}.yaml \
                    experiment_name={model}_al_ms_ft_{train_frac}_{dataset}_{p} \
                    datamodule=img_txt_mask_al/{dataset}.yaml \
                    prompt_type={p} \
                    train_frac={train_frac}\
                    model_name={model}\
                    datamodule.batch_size={batch_size} \
                    model.optimizer.lr={lr} \
                    tags='[{model}, {dataset}, finetune_al_ms, {p}]' \
                    output_masks_dir=output_masks/{model}/al_ms/al_ms_{train_frac}/{dataset}/test/{p}\
                    trainer.accelerator={accelerator} \
                    trainer.devices={devices} \
                    trainer.precision={precision}"

                if debugger:
                    command = f"{command} debug=default"

                # Log command in terminal
                print(f"RUNNING COMMAND \n{command}")

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
