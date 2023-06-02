import os.path
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm


def get_chexlocalize_mask_same_dir(
    mask_dir: str,
    output_dir: str,
):
    all_directories = glob(os.path.join(mask_dir, "*"))
    for directory in tqdm(all_directories):
        mask_paths = glob(os.path.join(directory, "*"))
        mask_name = os.path.basename(directory)
        for mask_path in mask_paths:
            mask_file = os.path.basename(mask_path)
            file_name, extension = mask_file.rsplit(".", 1)
            new_mask_file_name = f"{file_name}_{mask_name}.{extension}"
            output_path = os.path.join(output_dir, new_mask_file_name)
            # move the mask to new directory
            os.rename(mask_path, output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    get_chexlocalize_mask_same_dir(**vars(args))
