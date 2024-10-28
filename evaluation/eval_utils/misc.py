import itertools
import logging
import math
import types
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from mmseg.models import build_segmentor as build_segmentor_

from evaluation.depth.models import build_depther as build_depther_

# from mmdet.models import build_detector as build_detector_

logger = logging.getLogger(__name__)


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    # @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def build_segmentor(cfg, backbone_model):
    # make sure the backbone model is frozen;
    # we only do finetuning on the head
    for param in backbone_model.parameters():
        param.requires_grad = False

    model = build_segmentor_(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    if hasattr(backbone_model, "denoiser") and backbone_model.denoiser is not None:
        logger.info("Using the single-block denoiser")
        model.backbone.forward = lambda x: [backbone_model(x).permute(0, 3, 1, 2)] * 4
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.vit.model.patch_embed.patch_size[0])(x[0])
        )
    elif hasattr(backbone_model, "model") or hasattr(backbone_model, "patch_embed"):
        logger.info("Using the original or fine-tuned ViT as backbone.")
        if hasattr(backbone_model, "model"):
            backbone_model = backbone_model.model
        # check if the backbone_model has get_intermediate_layers method
        if not hasattr(backbone_model, "get_intermediate_layers"):
            if not hasattr(backbone_model, "forward_intermediates"):
                raise NotImplementedError(
                    "The backbone model does not have get_intermediate_layers method or "
                    "forward_intermediates method"
                )
            logger.info("Using forward_intermediates method as get_intermediate_layers")
            model.backbone.forward = partial(
                backbone_model.forward_intermediates,
                indices=cfg.model.backbone.out_indices,
                return_prefix_tokens=False,
                norm=True,
                output_fmt="NCHW",
            )
        else:
            model.backbone.forward = partial(
                backbone_model.get_intermediate_layers,
                n=cfg.model.backbone.out_indices,
                reshape=True,
            )
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_embed.patch_size[0])(x[0])
        )
    else:
        raise NotImplementedError
    # freeze all but the head
    for _, p in model.backbone.named_parameters():
        p.requires_grad = False

    model.init_weights()
    return model


def build_detector(cfg, backbone_model, enable_grad=False):
    # make sure the backbone model is frozen;
    # we only do finetuning on the head
    for param in backbone_model.parameters():
        param.requires_grad = enable_grad

    model = build_detector_(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))

    if hasattr(backbone_model, "denoiser") and backbone_model.denoiser is not None:
        logger.info("Using the single-block denoiser")
        model.backbone.forward = lambda x: backbone_model(x).permute(0, 3, 1, 2)
        # model.backbone.forward = lambda x: [
        #     backbone_model(x, return_channel_first=True)
        # ]
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.vit.model.patch_embed.patch_size[0])(x[0])
        )
    elif hasattr(backbone_model, "vit") or hasattr(backbone_model, "patch_embed"):
        logger.info("Using the original or fine-tuned ViT as backbone.")
        if hasattr(backbone_model, "vit"):
            backbone_model = backbone_model.vit.model
        model.backbone.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
            output_prefix_tokens=(not cfg.use_windowed_attn),
        )
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_embed.patch_size[0])(x[0])
        )
    else:
        raise NotImplementedError

    # freeze all but the head
    for _, p in model.backbone.named_parameters():
        p.requires_grad = enable_grad

    model.init_weights()
    return model


def build_depther(cfg, backbone_model):
    # make sure the backbone model is frozen;
    # we only do finetuning on the head
    for param in backbone_model.parameters():
        param.requires_grad = False

    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther_(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)
    if hasattr(backbone_model, "denoiser") and backbone_model.denoiser is not None:
        logger.info("Using the single-block denoiser")
        depther.backbone.forward = lambda x: [
            backbone_model(
                x,
                return_class_token=cfg.model.backbone.output_cls_token,
                norm=cfg.model.backbone.final_norm,
                return_channel_first=True,
            )
        ]
        depther.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.vit.model.patch_embed.patch_size[0])(x[0])
        )
    elif hasattr(backbone_model, "model") or hasattr(backbone_model, "patch_embed"):
        logger.info("Using the fine-tuned ViT or the original ViT as backbone.")
        if hasattr(backbone_model, "model"):
            backbone_model = backbone_model.model

        def format_output(x):
            return [(_x[0], _x[1][:, 0]) for _x in x]

        # check if the backbone_model has get_intermediate_layers method
        if not hasattr(backbone_model, "get_intermediate_layers"):
            if not hasattr(backbone_model, "forward_intermediates"):
                raise NotImplementedError(
                    "The backbone model does not have get_intermediate_layers method or "
                    "forward_intermediates method"
                )
            logger.info("Using forward_intermediates method as get_intermediate_layers")
            depther.backbone.forward = partial(
                backbone_model.forward_intermediates,
                indices=cfg.model.backbone.out_indices,
                return_prefix_tokens=cfg.model.backbone.output_cls_token,
                norm=cfg.model.backbone.final_norm,
                output_fmt="NCHW",
            )
        else:
            depther.backbone.forward = partial(
                backbone_model.get_intermediate_layers,
                n=cfg.model.backbone.out_indices,
                return_prefix_tokens=cfg.model.backbone.output_cls_token,
                norm=cfg.model.backbone.final_norm,
                reshape=True,
            )
        depther.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_embed.patch_size[0])(x[0])
        )
        depther.backbone.register_forward_hook(lambda _, x, y: format_output(y))
    else:
        raise NotImplementedError

    return depther
