import numpy as np
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class FeatureFormatBundle(object):
    """Feature formatting bundle.evaluation/configs/feature_vits14_voc2012_linear_config.py

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if "img" in results:
            img = results["img"]
            # if len(img.shape) < 3:
            #     img = np.expand_dims(img, -1)
            # img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results["img"] = DC(img.permute(2, 0, 1), stack=True)
        if "gt_semantic_seg" in results:
            # convert to long
            results["gt_semantic_seg"] = DC(
                to_tensor(results["gt_semantic_seg"][None, ...].astype(np.int64)),
                stack=True,
            )
        return results

    def __repr__(self):
        return self.__class__.__name__
