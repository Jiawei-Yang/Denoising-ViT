import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import Registry
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadFeaturesFromFile(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(
        self,
        file_client_args=dict(backend="disk"),
        imdecode_backend="pillow",
        feature_type="feature_type",
        ori_shape=None,
    ):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.feature_type = feature_type
        self.ori_shape = ori_shape

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        # import ipdb; ipdb.set_trace()

        if results.get("img_prefix") is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]
        # feature_filename = filename.replace(
        #     "JPEGImages", f"DenoisedFeatures/{self.feature_type}"
        # )
        feature_filename = filename  # for now i think this should be ok
        try:
            img_features = np.load(feature_filename).squeeze()
        except:
            print("Error loading features, using default")
            img_features = np.load(
                # "./work_dirs/vit_base_patch14_dinov2.lvd142m_s14/2007_000528.npy"
                "data/VOCdevkit/VOC2012/DenoisedFeatures/vit_base_patch14_dinov2.lvd142m_s14/2007_000528.npy"
            ).squeeze()

        # a bit of a hack, but we need to get the shape of the image
        if self.ori_shape is None:
            seg_filename = osp.join(results["seg_prefix"], results["img_info"]["ann"]["seg_map"])
            img_bytes = self.file_client.get(seg_filename)
            gt_semantic_seg = (
                mmcv.imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend)
                .squeeze()
                .astype(np.uint8)
            )
            results["ori_shape"] = gt_semantic_seg.shape
        else:
            results["ori_shape"] = self.ori_shape

        results["filename"] = feature_filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img_features
        results["img_shape"] = img_features.shape
        # results["ori_shape"] = img_features.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img_features.shape
        results["scale_factor"] = 1.0
        results["flip"] = False
        results["flip_direction"] = "horizontal"
        num_channels = 1 if len(img_features.shape) < 3 else img_features.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
