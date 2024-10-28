from typing import Tuple, Union

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
np.random.seed(0)


class SingleImageDataset:
    """
    Returns the randomly augmented view of a single image.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        base_transform: transforms.Compose = None,
        final_transform: transforms.Compose = None,
        num_views: int = 768,
    ):
        self.size = size
        self.base_transform = base_transform
        self.final_transform = final_transform
        self.num_views = num_views

    def set_image(self, img: Union[str, np.ndarray]):
        if not isinstance(img, np.ndarray):
            img = np.array(Image.open(img).convert("RGB"))
        self.image = img
        self.original_image = F.resize(
            self.base_transform(self.image),
            size=self.size,
            interpolation=Image.BICUBIC,
            antialias=True,
        )

    def __getitem__(self, index):
        # pixel coords are in the range of [0, 1]
        # pixel coords are in feature-resolution, i.e., (H/patch_size, W/patch_size)
        aug_view, pixel_coords = self.final_transform(self.original_image)
        return {
            "transformed_view": aug_view,
            "pixel_coords": pixel_coords,
            "full_image": self.original_image.clone(),
        }

    def __len__(self):
        return self.num_views
