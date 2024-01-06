import numpy as np
import torchvision.transforms.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
np.random.seed(0)


class SingleImageDataset(object):
    """
    Returns the randomly augmented view of a single image.
    """

    def __init__(
        self,
        img_pth="data/coco_examples/000000001773.jpg",
        size=(224, 224),
        base_transform=None,
        final_transform=None,
        num_iters=100000,
    ):
        self.image = Image.open(img_pth).convert("RGB")
        self.base_transform = base_transform
        self.final_transform = final_transform
        self.original_image = F.resize(
            self.base_transform(self.image), size=size, interpolation=Image.BICUBIC, antialias=True
        )
        self._iters = num_iters

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
        # iteration-based training
        return self._iters

    def update_iters(self, num_iters: int):
        self._iters = num_iters
