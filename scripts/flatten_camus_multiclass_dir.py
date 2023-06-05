import shutil
from pathlib import Path


def main(root_dir: Path, glob_pattern: str, unified_dir: Path):
    """Flatten a multiclass mask directory into a single directory.
    It converts `mask_?/*.png` to `unified_dir/*_?.png`

    Args:
        root_dir (Path): The root directory of input images
        glob_pattern (str): The glob pattern for the input images
        unified_dir (Path): The path to output directory

    Raises:
        ValueError: The mask directory name should end with a number
    """

    root_dir = Path(root_dir)

    unified_dir = Path(unified_dir)
    unified_dir.mkdir(parents=True, exist_ok=True)

    for path in root_dir.glob(glob_pattern):
        path_parent = str(path.parent)

        try:
            label_index = int(path_parent[-1])
        except ValueError:
            raise ValueError(
                f"Mask directory name should end with a number. Found {path_parent[-1]}"
            )

        new_path = unified_dir / (path.stem + f"_{label_index}" + path.suffix)

        shutil.copy(path, new_path)

        print("Copied", path, "to", new_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root-dir",
        type=Path,
        default="/mnt/Enterprise/PUBLIC_DATASETS/camus_database_png/training",
    )

    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="mask_[0-9]/*.png",
    )

    parser.add_argument(
        "--unified-dir",
        type=Path,
        default="/mnt/Enterprise/PUBLIC_DATASETS/camus_database_png/training/unified_mask_bin",
    )

    args = parser.parse_args()

    main(**vars(args))
