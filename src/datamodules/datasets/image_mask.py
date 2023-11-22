from typing import Any, Dict, Optional, Tuple, List
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import json
import random


class ImageMaskDataset(Dataset):
    r"""
    Image-Mask Dataset
    Args:
        images_dir (str): Path to the directory containing images
        masks_dir (str): Path to the directory containing masks
        caps_file (str): Path to the file containing captions
        class_name (str): Name of the class to filter the dataset
        img_size (int,int): Size of image. Defaults to (224, 224).
        img_transforms (T.Compose): Compose object containing transforms for images
        mask_transforms (T.Compose): Compose object containing transforms for masks
    Raises:
        ValueError: If data_num is of type float and is not in range [0, 1]
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        caps_file: Optional[str] = None,
        class_name: Optional[str | List[str]] = None,
        img_size: Tuple[int, int] = (224, 224),
        img_transforms: Optional[T.Compose] = None,
        mask_transforms: Optional[T.Compose] = None,
    ) -> None:
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.class_name = class_name

        with open(caps_file, "r") as f:
            self.img_captions = json.load(f)
            random.shuffle(self.img_captions)

        if self.class_name is not None and isinstance(self.class_name, str):
            self._filter(self.class_name)
        
        # Assign default img_transforms if no img_transforms is passed
        if self.img_transforms is None:
            self.img_transforms = T.Compose(
                [
                    T.Resize(size=img_size),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        # Assign default mask_transforms if no mask_transforms is passed
        if self.mask_transforms is None:
            self.mask_transforms = T.Compose(
                [
                    T.Resize(
                        size=img_size,
                        interpolation=T.InterpolationMode.NEAREST_EXACT,
                    ),
                    T.ToTensor(),
                ]
            )

    def _filter(self, class_name: str) -> None:
        self.img_captions = [
            cap for cap in self.img_captions if class_name in cap["mask_name"]
        ]

    def __len__(self) -> int:
        return len(self.img_captions)

    def __getitem__(self, index) -> Dict[str, Any]:
        cap = self.img_captions[index]
        image = Image.open(f"{self.images_dir}/{cap['img_name']}").convert("RGB")
        mask = Image.open(f"{self.masks_dir}/{cap['mask_name']}")

        # For multi-class segmentation only
        if self.class_name is not None and isinstance(self.class_name, list):
            for idx, class_ in enumerate(self.class_name):
                mask_file = cap["mask_name"]
                mask = Image.open(f"{self.masks_dir}/{cap['mask_name']}")
                if class_ in cap["mask_name"]:
                    mask = mask * (idx + 1)
                    break

        h, w = mask.height, mask.width

        image = self.img_transforms(image)
        mask = self.mask_transforms(mask)[:1] # ToTensor Gives 3-channeled mask

        return dict(
            pixel_values=image,
            mask=mask,
            mask_name=cap["mask_name"],
            height=h,
            width=w,
        )
