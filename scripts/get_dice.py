import argparse
from glob import glob
from statistics import mean

import cv2
import numpy as np
import pandas as pd
import torch
from monai.metrics import DiceMetric
from torchmetrics import Dice


def get_dice(predictions_path, labels_path, file_type):

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice = Dice(average="micro")

    prediction_files = sorted(glob(f'{predictions_path}*.{file_type}'))
    label_files = sorted(glob(f'{labels_path}*.{file_type}'))

    assert len(prediction_files) == len(label_files), "Predictions and labels number do not match"

    dice_vec = []

    for i in range(len(label_files)):
        prediction = cv2.imread(prediction_files[i], 0) // 128
        label = cv2.imread(label_files[i], 0) // 128
        prediction = torch.tensor(prediction, dtype=torch.int)
        #prediction = [torch.tensor(np.reshape(prediction, (1, prediction.shape[0], prediction.shape[1])) >= 128, dtype=torch.int)]
        label = torch.tensor(label, dtype=torch.int)
        # print(prediction.shape)
        # print(label.shape)
        #print(torch.unique(prediction==label))
        #print(prediction[200:220, 200:220])
        #print(label[300:320, 300:320])
        #label = [torch.tensor(np.reshape(label, (1, label.shape[0], label.shape[1])) >= 128, dtype=torch.int)]
        #dice_vec.append(dice_metric(prediction, label).numpy()[0][0])
        # prediction = torch.ones((100, 100), dtype=torch.int)
        # label = torch.zeros((100, 100), dtype=torch.int)
        # label[0][:34] = 1
        #dice_val = dice_metric(prediction, label).numpy()[0][0]
        dice_val = dice(prediction, label)
        dice_vec.append(dice_val.item())
        print(f'Dice entry recorded - {i} : {dice_val}')

    df = pd.DataFrame()
    df['file_path'] = prediction_files
    df['dice_score'] = dice_vec
    df['average_dice_score'] = mean(dice_vec)

    return dice_vec, df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--file_type", type=str, required=False)
    parser.add_argument("--csv_path", type=str, required=False)
    args = parser.parse_args()

    if args.file_type == None:
        args.file_type == "png"

    dice_vec, df = get_dice(args.predictions_path, args.labels_path, args.file_type)

    df.to_csv(f'{args.csv_path}')
    print(f'Dice score: {mean(dice_vec)}')

