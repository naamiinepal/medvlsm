from pathlib import Path
from typing import Union

StrPath = Union[str, Path]


def main(pretrain_folder: Path, checkpoint_name: StrPath, new_checkpoint_name: StrPath):
    """
    Converts a checkpoint trained with dataparallel on multiple GPUs to a checkpoint trained with a single GPU.
    This is useful when you want to use the checkpoint in a different machine with a different number of GPUs.

    Args:
        pretrain_folder (Path): Folder where the checkpoint is located.
        checkpoint_name (StrPath): Name of the checkpoint to convert.
        new_checkpoint_name (StrPath): Name of the new checkpoint.

    Example:
        python scripts/convert_cris_model.py --pretrain_folder pretrain --checkpoint_name cris_best.pth --new_checkpoint_name cris_best_single.pth
    """

    import torch

    checkpoint = torch.load(pretrain_folder / checkpoint_name, map_location="cpu")

    state_dict = checkpoint["state_dict"]

    starting_str = "module."

    assert (
        all([k.startswith(starting_str) for k in state_dict.keys()]),
        "Not all keys start with module. Probably a wrong checkpoint.",
    )

    start_offset = len(starting_str)

    new_state_dict = {k[start_offset:]: v for k, v in state_dict.items()}

    torch.save(new_state_dict, pretrain_folder / new_checkpoint_name, pickle_protocol=5)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--pretrain_folder", type=Path, default=Path("pretrained"))

    parser.add_argument("--checkpoint_name", type=str, default="cris_best.pth")

    parser.add_argument(
        "--new_checkpoint_name", type=str, default="cris.pt"
    )

    args = parser.parse_args()

    main(**vars(args))