from pathlib import Path

import pandas as pd

import seaborn as sns
from matplotlib.figure import Figure

from typing import List, Tuple, Union
from glob import glob

import json


def get_metrics(
    glob_pattern: str,
    prompts: Union[List[str], Tuple[str]],
    model_name: str,
    stage: str,
):
    metric_files = glob(glob_pattern)
    print(metric_files)

    assert len(metric_files) == len(
        prompts
    ), f"The number of prompts: {len(prompts)} and csv files: {len(metric_files)} should be equal"

    metric_files.sort()

    dataframes = []

    for file, p in zip(metric_files, prompts):
        df = pd.read_csv(file)
        df["prompt_type"] = p

        dataframes.append(df)

    concat_df = pd.concat(dataframes, ignore_index=True)

    concat_df["model_name"] = model_name
    concat_df["stage"] = stage

    return concat_df


def main(config_path: Path, output_dir: Path, metric: str, save_fmt: str = "png"):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        plots = json.load(f)

    plot_nums = len(plots)

    prompts_collection = [[f"p{i}" for i in range(p["prompt_number"])] for p in plots]

    glob_patterns = [plot["csv_glob"] for plot in plots]

    assert (
        len(glob_patterns) == plot_nums
    ), "The glob patterns must be equal to the prompt numbers"

    models = [plot["model_name"] for plot in plots]

    assert len(models) == plot_nums, "The models must be equal to the prompt numbers"

    stages = [plot["stage"] for plot in plots]

    assert len(stages) == plot_nums, "The stages must be equal to the prompt numbers"

    concat_df = pd.concat(
        [
            get_metrics(*args)
            for args in zip(glob_patterns, prompts_collection, models, stages)
        ],
        ignore_index=True,
    )

    metric_plot = sns.lineplot(
        data=concat_df,
        x="prompt_type",
        y=metric,
        hue="model_name",
        style="stage",
        markers=True,
    )

    models = "_".join(set(models))

    fig_name = f"{config_path.stem}_{metric}_{models}"

    fig: Figure = metric_plot.get_figure()
    fig.savefig(output_dir / f"{fig_name}.{save_fmt}", bbox_inches="tight")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the config file containing the glob patterns, model names and stages",
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
        help="The metric to plot. Must be one of iou, dice, surface_dice, hausdorff_distance",
    )

    parser.add_argument(
        "--save-fmt",
        type=str,
        default="png",
        help="The format to save the plots in. Must be one of png, pdf, svg",
    )

    args = parser.parse_args()

    main(**vars(args))
