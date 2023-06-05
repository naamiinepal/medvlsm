import glob
import cv2
from tqdm import tqdm


def resize_image_mask(image_path, mask_path, new_size):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, new_size)
    mask = cv2.resize(mask, new_size)
    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--new_size", type=int, nargs=2, default=[512, 512])

    args = parser.parse_args()

    image_paths = glob.glob(args.image_dir + "/*.jpg")
    mask_paths = glob.glob(args.mask_dir + "/*.png")

    for image_path, mask_path in tqdm(zip(image_paths, mask_paths)):
        resize_image_mask(image_path, mask_path, args.new_size)
