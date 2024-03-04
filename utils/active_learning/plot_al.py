
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

fracs = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
train_dataset ='kvasir_polyp'
test_datasets = ['clinicdb_polyp']
model = 'cris'
mechanisms = ['rs', 'ms']
dice_dict = {} 
iou_dict = {}

for test_ds in test_datasets:
    dice_list =[]
    iou_list=[]
    for mechanism in mechanisms:
                
        for frac in fracs:

            if test_ds == train_dataset:

                csv_path = f"/mnt/Enterprise2/kanchan/medvlsm/output_masks/{model}/al_{mechanism}/al_{mechanism}_{frac}/{train_dataset}/test/random.csv"
            else:
                csv_path = f"/mnt/Enterprise2/kanchan/medvlsm/output_masks/{model}/al_{mechanism}_cross/al_{mechanism}_{frac}/trained_on_{train_dataset}/tested_on_{test_ds}/test/random.csv"

            df = pd.read_csv(csv_path)

            dice=np.nanmean(df["dice"])
            iou=np.nanmean(df["iou"])

            dice_list.append([frac, mechanism, dice])
            iou_list.append([frac, mechanism, iou])

    dice_dict[f"{test_ds}"]=pd.DataFrame(columns=['sampling_frac', 'method', 'dice'], data=dice_list)
    iou_dict[f"{test_ds}"]=pd.DataFrame(columns=['sampling_frac', 'method', 'iou'], data=iou_list)



print(dice_dict)
sns.set_theme(style="whitegrid")

metric_plot = sns.lineplot(
        data=dice_dict["clinicdb_polyp"],
        x="sampling_frac",
        y="dice",
        hue="method",
        markers=True,
        legend="auto")

fig: Figure = metric_plot.get_figure()

fig.savefig(
        f"line_clinicdb_{model}.png", bbox_inches="tight", pad_inches=0.05
    )
