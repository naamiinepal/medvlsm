from pathlib import Path


def main(
    root_dir: Path, glob_pattern: str, unified_dir: Path, verbose: bool, dryrun: bool
):
    """Flatten a multiclass mask directory into a single directory.

    Args:
        root_dir (Path): The root directory of input images
        glob_pattern (str): The glob pattern for the input images
        unified_dir (Path): The path to output directory
        verbose (bool): Whether to print the copied paths
    """

    if dryrun:
        print("Dry run mode, no files will be copied")
    else:
        import shutil

    unified_dir.mkdir(parents=True, exist_ok=True)

    for path in root_dir.glob(glob_pattern):
        us_type = path.parents[1].name

        new_path = unified_dir / (path.stem + f"_{us_type}" + path.suffix)

        if not dryrun:
            shutil.copy(path, new_path)

        if dryrun or verbose:
            print("Copied", path, "to", new_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--root-dir", type=Path, required=True)

    parser.add_argument("--glob-pattern", type=str, required=True)

    parser.add_argument("--unified-dir", type=Path, required=True)

    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument(
        "--dryrun", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()

    main(**vars(args))
