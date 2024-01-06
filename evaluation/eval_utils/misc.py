import itertools
import math
import types
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.models import build_segmentor as build_segmentor_

from evaluation.depth.models import build_depther


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
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])
        )
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
    if hasattr(backbone_model, "patch_embed"):
        model.backbone.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
        )
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_embed.patch_size[0])(x[0])
        )
    elif hasattr(backbone_model, "denoiser"):
        print("hi i'm in denoiser")
        model.backbone.forward = lambda x: [backbone_model(x).permute(0, 3, 1, 2)] * 4
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(
                backbone_model.vit.model.patch_embed.patch_size[0]
            )(x[0])
        )
    else:

        def forward(x):
            return [x] * 4

        model.backbone.forward = forward

    # freeze all but the head
    for _, p in model.backbone.named_parameters():
        p.requires_grad = False

    model.init_weights()
    return model


def create_depther(cfg, backbone_model):
    # make sure the backbone model is frozen;
    # we only do finetuning on the head
    for param in backbone_model.parameters():
        param.requires_grad = False

    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    if hasattr(backbone_model, "patch_size"):
        # fb model
        depther.backbone.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
            return_class_token=cfg.model.backbone.output_cls_token,
            norm=cfg.model.backbone.final_norm,
        )
        depther.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )
    elif hasattr(backbone_model, "patch_embed"):
        depther.backbone.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
            return_class_token=cfg.model.backbone.output_cls_token,
            norm=cfg.model.backbone.final_norm,
        )
        depther.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_embed.patch_size[0])(x[0])
        )

    elif hasattr(backbone_model, "denoiser"):
        depther.backbone.forward = lambda x: [
            backbone_model(
                x,
                return_class_token=cfg.model.backbone.output_cls_token,
                norm=cfg.model.backbone.final_norm,
                return_channel_first=True,
            )
        ]
        depther.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(
                backbone_model.vit.model.patch_embed.patch_size[0]
            )(x[0])
        )

    else:

        def forward(x):
            return [x]

        depther.backbone.forward = forward

    return depther
