import argparse
import os
from glob import glob
from shutil import copyfile


def rename_files(input_image_dir, input_mask_dir, output_image_dir, output_mask_dir, image_format='jpg', mask_format='png', initial_count=0):

    if not os.path.exists(args.output_image_dir):
        os.makedirs(args.output_image_dir)

    if not os.path.exists(args.output_mask_dir):
        os.makedirs(args.output_mask_dir)

    image_files = sorted(glob(f'{input_image_dir}*.*'))
    mask_files = sorted(glob(f'{input_mask_dir}*.*'))

    assert len(image_files) == len(mask_files), "Image and mask numbers do not match"

    if initial_count != None:
        count = initial_count
    else:
        count = 0

    for idx in range(len(image_files)):
        copyfile(image_files[idx], f'{output_image_dir}{count}.{image_format}')
        copyfile(mask_files[idx], f'{output_mask_dir}{count}.{mask_format}')
        count += 1

    return 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_dir", type=str, required=True)
    parser.add_argument("--input_mask_dir", type=str, required=True)
    parser.add_argument("--output_image_dir", type=str, required=True)
    parser.add_argument("--output_mask_dir", type=str, required=True)
    parser.add_argument("--image_format", type=str, required=False)
    parser.add_argument("--mask_format", type=str, required=False)
    parser.add_argument("--initial_count", type=int, required=False)
    args = parser.parse_args()

    rename_files(args.input_image_dir, args.input_mask_dir, args.output_image_dir, args.output_mask_dir, args.image_format, args.mask_format, args.initial_count)
    print('Rename completed')
