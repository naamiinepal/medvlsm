import concurrent.futures
import os.path
from argparse import ArgumentParser, Namespace
from glob import glob

import cv2
import numpy as np
from albumentations.augmentations.geometric.resize import SmallestMaxSize
from PIL import Image
from tqdm import tqdm


def resize_image(image_path: str, size: int):
    """
    Resize the image without changing the aspect ratio.

    Args:
        image_path (str): path to the image
        size (int): shortest size of the image and mask to be resized to
    """
    image = np.array(Image.open(image_path))

    image = SmallestMaxSize(max_size=size, interpolation=cv2.INTER_CUBIC)(image=image)[
        "image"
    ]
    return image


def main(args: Namespace):
    image_paths = glob(os.path.join(args.image_dir, args.image_pattern))

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # concat the image and mask paths
    all_paths = image_paths
    # resize the images and masks
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = []
        for image_path in tqdm(all_paths):
            futures.append(executor.submit(resize_image, image_path, args.size))

        for future, image_path in tqdm(zip(futures, all_paths)):
            image = future.result()
            image_name = os.path.basename(image_path)
            image_path = os.path.join(args.output_dir, image_name)
            Image.fromarray(image).save(image_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-pattern", type=str, default="*.png")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
