from pathlib import Path

import cv2
import numpy as np
from features_from_img import get_mask_decription
from tqdm import tqdm

ROOT_DIR = Path("/mnt/Enterprise/PUBLIC_DATASETS/us_images/br-usg/masks/")

glob_pattern = "*.png"

outputs = []

mask_paths = tuple(ROOT_DIR.glob(glob_pattern))

for mask_path in tqdm(mask_paths):
    mask_ori = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # sizes, positions, numbers
    output = get_mask_decription(mask_ori)

    outputs.append(output)

outputs = np.array(outputs)

sizes = outputs[:, 0]
positions = outputs[:, 1]
numbers = outputs[:, 2]
