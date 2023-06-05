from typing import Any, Optional, Union
import SimpleITK as sitk
from SimpleITK.extra import PathType as SiTKPathType
from pathlib import Path
import concurrent.futures
import numpy as np
import skimage as sk
from tqdm import tqdm

StrPath = Union[str, Path]


def load_and_save(
    filename: SiTKPathType,
    output_filename: SiTKPathType,
    mask_index: Optional[int],
    resample_size: Optional[int],
    order: int,
):
    """Load a SimpleITK image and save it with compression level of 9.

    Args:
        filename (SiTKPathType): The path to the image
        output_filename (SiTKPathType): The path to save the image
        mask_index (Optional[int]): The mask index to convert multiclass to binary mask.
    """

    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    if mask_index is not None:
        # Extracts the mask from the image
        # itkimage = sitk.BinaryThreshold(
        #     itkimage,
        #     lowerThreshold=mask_index,
        #     upperThreshold=mask_index,
        #     insideValue=255,
        # )
        itkimage = (itkimage == mask_index) * 255

        # Overwride the order for binary mask
        order = 0

    if resample_size is not None:
        # Resample the image to a specified size
        np_img = sitk.GetArrayFromImage(itkimage).squeeze()

        np_img = sk.transform.resize(
            np_img,
            (resample_size, resample_size),
            order=order,
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)

        itkimage = sitk.GetImageFromArray(np_img)

    # Save with compression level of 9
    return sitk.WriteImage(
        itkimage, output_filename, useCompression=True, compressionLevel=9
    )


def main(
    image_root: Path,
    glob_pattern: str,
    out_dir: Path,
    out_ext: str,
    mask_index: Optional[int],
    max_workers: Optional[int],
    resample_size: Optional[int],
    order: int,
):
    """Convert a directory of SimpleITK images to a directory of images with a specified extension.

    Args:
        image_root (Path): The root directory of input images
        glob_pattern (str): The glob pattern for the input images
        out_dir (Path): The path to output directory
        out_ext (str): The extension of output images
        mask_index (Optional[int], optional): The mask index to convert multiclass to binary mask. Defaults to None.
        max_workers (Optional[int], optional): The max workers for multiprocessing. Defaults to None.
    """
    # Create the output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for image_path in image_root.rglob(glob_pattern):
            image_name = image_path.stem
            output_filename = out_dir / (image_name + out_ext)
            f = executor.submit(
                load_and_save,
                image_path,
                output_filename,
                mask_index,
                resample_size,
                order,
            )
            futures[f] = (image_path, output_filename)

        pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures))

        for future in pbar:
            img_path, out_fname = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{img_path} generated an exception: {exc}")
            else:
                pbar.set_postfix({"in": img_path.name, "out": out_fname.name})


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="The root directory of input images",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        required=True,
        help="The glob pattern for the input images",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--out-ext", type=str, default=".png", help="The extension of output images"
    )
    parser.add_argument(
        "--mask-index",
        type=int,
        default=None,
        help="Mask index to convert multiclass to binary mask",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None, help="Max workers for multiprocessing"
    )
    parser.add_argument(
        "--resample-size",
        type=int,
        default=None,
        help="Resample size for resampling",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="Interpolation order for resampling",
    )

    args = parser.parse_args()

    main(**vars(args))
