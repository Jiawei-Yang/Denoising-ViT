from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F


class RandomResizedCropFlip(torchvision.transforms.RandomResizedCrop):
    """
    Modified from torchvision.transforms.RandomResizedCrop.
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=F.InterpolationMode.BICUBIC,
        antialias: Optional[bool] = True,
        horizontal_flip=True,
        patch_size: Optional[int] = 14,
        stride: Optional[int] = 14,
    ):
        super().__init__(
            size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
            antialias=antialias,
        )
        self.patch_size = patch_size
        self.stride = stride
        self.horizontal_flip = horizontal_flip
        # Calculate the shape of feature map
        self.h_patches = (self.size[0] - self.patch_size) // self.stride + 1
        self.w_patches = (self.size[1] - self.patch_size) // self.stride + 1

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        # top, left, height, width
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        _, height, width = F.get_dimensions(img)
        transformed_view = F.resized_crop(
            img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )

        # Calculate the coordinates of the transformed view in the original image
        norm_i, norm_j = i / float(height), j / float(width)
        norm_h, norm_w = h / float(height), w / float(width)

        # Calculate evenly spaced values for x and y under feature-resolution
        linspace_y = torch.linspace(norm_i, norm_i + norm_h, self.h_patches)
        linspace_x = torch.linspace(norm_j, norm_j + norm_w, self.w_patches)

        # Create a grid of x and y indices using torch.meshgrid
        grid_y, grid_x = torch.meshgrid(linspace_y, linspace_x, indexing="ij")

        # The meshgrid has created the normalized coordinates for each patch
        normalized_patch_coordinates = torch.stack([grid_x, grid_y], dim=-1)

        # Apply horizontal flip randomly
        if self.horizontal_flip and np.random.random() < 0.5:
            transformed_view = F.hflip(transformed_view)
            normalized_patch_coordinates[:, :, 0] = (
                normalized_patch_coordinates[:, :, 0].max() - normalized_patch_coordinates[:, :, 0]
            ) + normalized_patch_coordinates[:, :, 0].min()

        # the coordinates are in ##feature-resolution##, not pixel-resolution
        return (transformed_view, normalized_patch_coordinates)
