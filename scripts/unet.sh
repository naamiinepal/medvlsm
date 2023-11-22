#!/bin/bash

######################################
# FULL-TRAINING SEGMENTATION CONFIGS #
######################################

#!/bin/bash

# Model configs
train_models=("unet")

# # Dataset configs
# datasets=("kvasir_polyp" "clinicdb_polyp" "bkai_polyp" "isic" "dfu")

# for model in "${train_models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         python src/train.py \
#             experiment=${model}.yaml \
#             experiment_name=${model}_${dataset} \
#             datamodule=img_mask_${dataset}.yaml \
#             tags="[${model}, ${dataset}]" \
#             output_masks_dir=output_masks/${model}/${dataset}/
#     done
# done

# datasets=("camus" "busi" "chexlocalize")

# declare -A class_configs
# class_configs[camus]="mask_1 mask_2 mask_3"
# class_configs[busi]="benign malignant normal"
# class_configs[chexlocalize]="airspace_opacity cardiomegaly enlarged_cardiomediastinum support_devices edema consolidation pleural_effusion pneumothorax atelectasis lung_lesion"

# for model in "${train_models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         classes="${class_configs[$dataset]}"
#         for class in ${classes[@]}; do
#             python src/train.py \
#                 experiment=${model}.yaml \
#                 experiment_name=${model}_${dataset}_${class} \
#                 class_name=${class} \
#                 datamodule=img_mask_${dataset}.yaml \
#                 tags="[${model}, ${dataset}, ${class}]" \
#                 output_masks_dir=output_masks/${model}/${dataset}/${class}/
#         done
#     done
# done



# datasets=("camus")

# declare -A class_configs
# class_configs[camus]="mask_1 mask_2 mask_3"
# class_configs[busi]="benign malignant normal"
# class_configs[chexlocalize]="airspace_opacity cardiomegaly enlarged_cardiomediastinum support_devices edema consolidation pleural_effusion pneumothorax atelectasis lung_lesion"

# for model in "${train_models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         python src/train.py \
#             experiment=${model}.yaml \
#             experiment_name=${model}_${dataset}_multi_class \
#             model.net.classes=4 \
#             model.multi_class=true \
#             datamodule=img_mask_${dataset}.yaml \
#             tags="[${model}, ${dataset}, multi_class]" \
#             output_masks_dir=output_masks/${model}/${dataset}/multi_class/
#     done
# done


python src/train.py \
    experiment=unet.yaml \
    experiment_name=unet_mixed_all_tuned_dice_loss \
    model.net.classes=17 \
    model.log_output_masks=false \
    datamodule=img_mask_mixed_all.yaml \
    output_masks_dir=output_masks/unet/mixed_all/
