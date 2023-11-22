from typing import Any, Dict, Literal
from torch.utils.data import Dataset
from pathlib import Path
from ...utils import CLASS_NAMES
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import torch


class ImageMaskMixedDataset(Dataset):
    r"""
    Image-Mask Dataset
    Args:
        dataset_dir (str): Path to the directory containing images
        dataset_name (str): Name of the dataset
        split (str): Split of the dataset
        img_size (int): Size of the image to resize for default transforms
    """

    def __init__(
        self,
        dataset_dir: str,
        dataset_name: str,
        split: Literal["train", "val", "test"],
        img_size: int = 224,
        # transforms: Optional[A.Compose] = None,
    ) -> None:
        super().__init__()
        self.class_names = CLASS_NAMES
        self.dataset_name = dataset_name
        self.images_dir = Path(dataset_dir) / dataset_name / split / "images"
        self.masks_dir = Path(dataset_dir) / dataset_name / split / "masks"

        self.images = list(self.images_dir.glob("*"))
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        self.img_size = img_size
        self.transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # default values for imagenet
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.NEAREST_EXACT,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.images[index]
        image = read_image(str(image_path), mode=ImageReadMode.RGB).float()
        image_stem = image_path.stem
        masks = dict()
        mask_name = None
        mask_h, mask_w = None, None
        for class_path in self.masks_dir.glob("*"):
            class_name = class_path.stem
            if class_name in self.class_names:
                mask_paths = list((self.masks_dir / class_name).glob(f"{image_stem}*"))
                if len(mask_paths) == 0:
                    continue
                mask_path = mask_paths[0]
                mask_name = mask_path.name
                mask = read_image(str(mask_path), mode=ImageReadMode.GRAY)
                if mask_h is None or mask_w is None:
                    mask_h, mask_w = mask.shape[1:]
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0),
                    (self.img_size, self.img_size),
                    mode="nearest-exact",
                ).squeeze()
                mask = (mask > 127).long()
                masks[class_name] = mask
            else:
                ValueError(f"Class name {class_name} not in {self.class_names}")

        h, w = image.shape[1:]
        mask = torch.zeros((len(self.class_names), self.img_size, self.img_size))
        for class_name, msk in masks.items():
            mask[self.class_names.index(class_name)] = msk

        image = self.transforms(image)

        return dict(
            pixel_values=image,
            mask=mask,
            mask_name=mask_name,
            height=mask_h,
            width=mask_w,
            dataset=self.dataset_name,
        )
