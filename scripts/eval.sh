#!/bin/bash


# Map dataet to classes
declare -A class_configs
class_configs[kvasir_polyp]="polyp"
class_configs[clinicdb_polyp]="polyp"
class_configs[bkai_polyp]="polyp"
class_configs[cvc300_polyp]="polyp"
class_configs[cvc_colondb_polyp]="polyp"
class_configs[etis_polyp]="polyp"
class_configs[isic]="skin_cancer"
class_configs[dfu]="foot_ulcer"
class_configs[camus]="myocardium ventricle atrium"
class_configs[busi]="tumor"
class_configs[chexlocalize]="airspace_opacity atelectasis cardiomegaly consolidation edema enlarged_cardiomediastinum lung_lesion pleural_effusion pneumothorax support_devices"


# source ./scripts/configs.sh

# for model in "${train_models}"; do


# for model in "${train_models[@]}"; do
#     for dataset in "${datasets_finetune[@]}"; do
#         prompts="${prompts_configs[$dataset]}"
#         for prompt in ${prompts[@]}; do
#             IFS='-'
#             read -ra prompt <<< "$prompt"
#             python evaluate/eval_metrics.py \
#                 --seg_path=output_masks/${model}/finetune/${dataset}/${prompt}/ \
#                 --gt_path=data/${dataset}/masks \
#                 --csv_path=output_masks/${model}/finetune/${dataset}/${prompt}.csv
#             echo ====================================================================================================
#             echo ''
#             IFS=' '
#         done
#         brea
#     done
# done

train_models=("unet" "unetpp" "deeplabv3plus")
# datasets=(airspace_opacity atelectasis cardiomegaly consolidation edema enlarged_cardiomediastinum lung_lesion pleural_effusion pneumothorax support_devices)
datasets=("kvasir_polyp" "clinicdb_polyp" "bkai_polyp" "cvc300_polyp" "cvc_colondb_polyp" "etis_polyp")
prompts=("p0" "p6" "p9" "random")
for model in "${train_models[@]}"; do
    for dataset in "${datasets[@]}"; do
    # echo ${class_config[$dataset][@]}
        for prompt in "${prompts[@]}"; do
            python evaluate/eval_metrics.py \
                --seg_path=output_masks/${model}/mixed_polyp/${dataset}/${prompt} \
                --gt_path=data/${dataset}/masks/ \
                --csv_path=output_masks/${model}/mixed_polyp/${dataset}/${prompt}.csv
            echo ===============================================================
            echo ' '
        done
    done
done

datasets=("kvasir_polyp" "clinicdb_polyp" "bkai_polyp" "cvc300_polyp" "cvc_colondb_polyp" "etis_polyp")

# for model in "${train_models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#     # echo ${class_config[$dataset][@]}
#         if [ $dataset != "camus" ]; then
#             continue
#         fi
#         python scripts/eval_metrics.py \
#             --seg-path=output_masks/${model}/mixed_all/${dataset}/ \
#             --seg-glob-pattern "**/*" \
#             --gt-path=data/${dataset}/masks/test \
#             --gt-glob-pattern "**/*" \
#             --csv-path=output_masks/${model}/mixed_all/${dataset}.csv
#         echo ===============================================================
#         echo ' '
#     done
# done

# for model in "${train_models[@]}"; do
#     for dataset in "${datasets_finetune[@]}"; do
#         if [ $dataset != "camus" ]; then
#             continue
#         fi
#         prompts="${prompts_configs[$dataset]}"
#         for prompt in ${prompts[@]}; do
#             IFS='-' 
#             read -ra prompt <<< "$prompt"
#             python scripts/eval_metrics.py \
#                 --seg-path output_masks/${model}/finetune/${dataset}/${prompt}/ \
#                 --seg-glob-pattern "**/*" \
#                 --gt-path data/${dataset}/masks/test/ \
#                 --gt-glob-pattern "**/*" \
#                 --csv-path output_masks/${model}/finetune/${dataset}/${prompt}.csv
#                 echo "===================================================================================================="
#                 echo ""
#             IFS=' '
#         done
#     done
# done