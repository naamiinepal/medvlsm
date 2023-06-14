import json
import os
from glob import glob
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

np.random.seed(444)


def get_box_plot_df(
    glob_pattern: str, prompts: Union[List[str], Tuple[str]], colum_name: str
):
    metric_files = glob(glob_pattern)

    metric_files.sort()

    assert len(metric_files) == len(
        prompts
    ), f"The number of prompts: {len(prompts)} and csv files: {len(metric_files)} should be equal"

    metric_files.sort()

    df_all = pd.DataFrame()

    for file, p in zip(metric_files, prompts):
        df_all[p] = pd.read_csv(file)[colum_name]

    return df_all


def main(
    config_path: Path,
    output_dir: Path,
    metric: str,
    save_fmt: str,
    num_columns_plot: int,
    yrange: Tuple[float, float],
    fig_size: Tuple[float, float],
    x_label: str,
    y_label: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        plots = json.load(f)

    plot_nums = len(plots)

    prompts_collection = [[f"p{i}" for i in range(p["prompt_number"])] for p in plots]

    glob_patterns = [plot["csv_glob"] for plot in plots]
    assert (
        len(glob_patterns) == plot_nums
    ), "The glob patterns must be equal to the number of plots"

    models = [plot["model_name"] for plot in plots]
    assert len(models) == plot_nums, "The models must be equal to number of plots"

    datasets = [plot["dataset"] for plot in plots]
    assert len(datasets) == plot_nums, "The datasets must be equal to number of plots"

    sns.set(style="darkgrid")
    sns.set(font_scale=1.2)

    fig = plt.figure()
    num_rows_plot = ceil(plot_nums / num_columns_plot)
    fig, axes = plt.subplots(
        num_rows_plot,
        num_columns_plot,
        figsize=tuple(fig_size),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    if plot_nums == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()

    fig.supxlabel(x_label, fontsize=20)
    fig.supylabel(y_label, fontsize=20)

    for i, (glob_pattern, prompts, model, dataset) in enumerate(
        zip(glob_patterns, prompts_collection, models, datasets)
    ):
        axes[i].set_axis_on()
        df_box = get_box_plot_df(glob_pattern, prompts, metric)
        sns.boxplot(df_box, ax=axes[i]).set(title=f"Dataset: {dataset}", ylim=yrange)

    fig.savefig(f"{output_dir}_{model}.{save_fmt}", bbox_inches="tight")


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser()

    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the config file containing the glob patterns",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the output directory where the plots will be saved",
    )

    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="The metric to plot. Must be one of the columns in the csv files found in glob pattern above",
    ),

    parser.add_argument(
        "--save-fmt",
        type=str,
        default="png",
        help="The format to save the plots in. Must be one of png, pdf, svg",
    ),
    parser.add_argument(
        "--num-columns-plot",
        type=int,
        required=True,
        help="The number of columns in grid of the plot",
    ),

    parser.add_argument(
        "--yrange",
        type=float,
        nargs=2,
        default=(-100, 100),
        help="The range of the y-axis",
    ),
    parser.add_argument(
        "--fig-size",
        type=int,
        nargs=2,
        default=(15, 7),
        help="The size of figure",
    )

    parser.add_argument(
        "--x-label",
        type=str,
        default="prompt type",
        help="The x axis label",
    )
    parser.add_argument(
        "--y-label",
        type=str,
        default="zero-shot dice - all pixels white dice",
        help="The y axis label",
    )

    args = parser.parse_args()

    main(**vars(args))
