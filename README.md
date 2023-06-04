# VLM-SEG-2023
**SCRIPTS TO RUN CRIS**

1. Prepare Dataset in coco format with annotations
Sample at /mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/datasets/anns/kvasir_polyp_80_10_10
In the key "prompts" of each of the annotation, there should be key value pair of prompts of each type (P0, P1, ...., Pn for n types of prompts). In case a list of multiple prompts is given in a prompt type, one prompt is randomly sampled and sent during training

2. Prepare LMDB
    Use the scipt CRIS.pytorch/prepare_datasets_all.sh, by editing necessary paths

3. Edit path arguments and prompt type in config
    Data: Paths to fill are commented
    Train: Chnage the following args 
        output_folder
        prompt_type
4. Run scripts
    a. For inference
    Use the script CRIS.pytorch/test.sh as sample and edit necessary paths
    b. For finetunning
    Use the script CRIS.pytorch/finetune.sh as sample and edit necessary paths

