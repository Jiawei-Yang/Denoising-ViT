import math
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmseg.datasets.builder import PIPELINES

from evaluation.depth.ops import resize


@PIPELINES.register_module()
class FeatureResize(object):
    """Resize features.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(
        self,
        img_scale=None,
        multiscale_mode="range",
        ratio_range=None,
        keep_ratio=True,
        feature_stride: int = 14,
        feature_patch_size: int = 14,
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.feature_stride = feature_stride
        self.feature_patch_size = feature_patch_size

        def get_pad(size, patch_size):
            new_size = math.ceil(size / patch_size) * patch_size
            return new_size

        # if self.img_scale is not None:
        #     self.feature_scale = [
        #         (
        #             (get_pad(img_scale[0], feature_patch_size) - feature_patch_size)
        #             // feature_stride
        #             + 1,
        #             (get_pad(img_scale[1], feature_patch_size) - feature_patch_size)
        #             // feature_stride
        #             + 1,
        #         )
        #         for img_scale in self.img_scale
        #     ]
        #     print("feature scale", self.feature_scale)
        # else:
        #     self.feature_scale = None

    def _resize_feature(self, feat_img, size=None, scale_factor=None):
        return (
            F.interpolate(
                feat_img.unsqueeze(0).permute(0, 3, 1, 2),
                size=size,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=None,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

    def _scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        scale, scale_idx = self.img_scale[0], 0
        # feature_scale = self.feature_scale[0]

        results["scale"] = scale
        results["scale_idx"] = scale_idx
        # results["feature_scale"] = feature_scale

    def _resize_img(self, results):
        img = results["img"]
        h, w = img.shape[:2]

        w_scale, h_scale = 1, 1

        # # rescaled_img = imresize(
        # # img, new_size, interpolation=interpolation, backend=backend)
        # """Resize images with ``results['scale']``."""
        # if self.keep_ratio:
        #     new_size, scale_factor = mmcv.rescale_size(
        #         (w, h), results["feature_scale"], return_scale=True
        #     )
        #     img = self._resize_feature(
        #         img,
        #         size=new_size,
        #         # scale_factor=scale_factor,
        #     )
        # else:
        #     img = resize(
        #         img,
        #         size=results["feature_scale"],
        #     )

        # new_h, new_w = img.shape[:2]
        # w_scale = new_w / w
        # h_scale = new_h / h
        # print(
        #     "img shape",
        #     img.shape,
        #     "scale factor",
        #     scale_factor,
        #     "feature scale factor",
        #     scale_factor,
        #     "keep ratio",
        #     self.keep_ratio,
        #     "new size",
        #     new_size,
        # )

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results["img"] = img
        results["img_shape"] = img.shape  # (518, 518)
        results["pad_shape"] = img.shape  # (518, 518)  # in case that there is no padding
        results["scale_factor"] = scale_factor
        # results["feature_scale_factor"] = scale_factor
        results["keep_ratio"] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get("seg_fields", []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(results[key], results["scale"], interpolation="nearest")
            else:
                gt_seg = mmcv.imresize(results[key], results["scale"], interpolation="nearest")
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, depth estimation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if "scale" not in results:
            self._scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(img_scale={self.img_scale}, "
            f"multiscale_mode={self.multiscale_mode}, "
            f"ratio_range={self.ratio_range}, "
            f"keep_ratio={self.keep_ratio})"
        )
        return repr_str
