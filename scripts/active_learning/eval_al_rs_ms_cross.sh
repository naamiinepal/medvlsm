#!/bin/bash

mechanisms=("ms" "rs")
fracs=(1.0)
train_datasets=("kvasir_polyp")
test_datasets=("bkai_polyp" "clinicdb_polyp")
log_file="eval_metrics_active_learning_v2.log"
models=("cris" "clipseg")
p="random"
root_dir=#abs dir of the medvlsm repo
echo ========================= start ================================ >> ${log_file}
for model in ${models[@]}; do
    echo $model >> $log_file
    for mechanism in ${mechanisms[@]}; do
        echo $mechanism >> $log_file
        for frac in ${fracs[@]}; do
            echo $frac >> $log_file
            for ds in ${train_datasets[@]}; do
                echo ================================================================= >> ${log_file}
                echo Trained on: $ds >> $log_file
                python3 utils/eval_metrics.py \
                --seg_path=${model}/output_masks/${model}/al_${mechanism}/al_${mechanism}_${frac}/${ds}/test/${p} \
                --gt_path=${model}/datasets/${ds}/masks/ \
                --csv_path=${model}/output_masks/${model}/al_${mechanism}/al_${mechanism}_${frac}/${ds}/test/${p}.csv >> $log_file    
                echo Completed testing on same dataset >> $log_file        

                for test_ds in ${test_datasets[@]}; do
                    python3 utils/eval_metrics.py \
                    --seg_path=${model}/output_masks/${model}/al_${mechanism}_cross/al_${mechanism}_${frac}/trained_on_${ds}/tested_on_${test_ds}/test/${p} \
                    --gt_path=${model}/datasets/${test_ds}/masks/ \
                    --csv_path=${model}/output_masks/${model}/al_${mechanism}_cross/al_${mechanism}_${frac}/trained_on_${ds}/tested_on_${test_ds}/test/${p}.csv >> $log_file
                    echo Completed testing on dataset $test_ds >> $log_file        
                done

                echo ================================================================= >> ${log_file}
            done
        done
    done
done
echo ========================= end ================================ >> ${log_file}


