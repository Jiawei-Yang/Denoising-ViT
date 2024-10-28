import logging
import re
import types
from typing import List, Optional, Tuple, Union

import timm
import timm.data
import torch
from timm.models.eva import Eva
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision import transforms

# We have played with these models, Feel free to add more models to the list.
MODEL_LIST = [
    # DINOv1
    "vit_small_patch8_224.dino",
    "vit_small_patch16_224.dino",
    "vit_base_patch8_224.dino",
    "vit_base_patch16_224.dino",
    # DINOv2
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
    # DINOv2 + register
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_giant_patch14_reg4_dinov2.lvd142m",
    # MAE
    "vit_base_patch16_224.mae",
    "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    # CLIP
    "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
    "vit_base_patch16_clip_224.openai",
    # EVA
    "eva02_base_patch16_clip_224.merged2b",
    # DEiT-III
    "deit3_base_patch16_224.fb_in1k",
    # Auto-auged supervised ViT:
    "vit_base_patch16_384.augreg_in21k_ft_in1k",
    # commented out for simplicity. Do not use these models for now
    # it's a bit annoying to hack the intermediate layers for these models
    # however, in our informal early experiments, these models all exhibit
    # similar artifacts as the models above.
    # the artifacts in SAM are similar to the ones in MAE, while the
    # artifacts in iJEPGA are similar to the ones in DeiT.
    # SAM
    # "samvit_base_patch16.sa1b",
    # "samvit_large_patch16.sa1b",
    # "samvit_huge_patch16.sa1b",
    # ijepga
    # "vit_huge_patch14_224_ijepa.in1k",
]


class PretrainedViTWrapper(nn.Module):
    def __init__(
        self,
        model_identifier: str = "vit_base_patch14_dinov2.lvd142m",
        stride: int = 7,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        **kwargs,
    ):
        super().__init__()
        # comment out the following line to test the models not in the list
        assert model_identifier in MODEL_LIST, f"Model type {model_identifier} not tested yet."
        self.model_identifier = model_identifier
        self.stride = stride
        self.patch_size = int(re.search(r"patch(\d+)", model_identifier).group(1))
        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.model, self.transformation = self.create_model(model_identifier, **kwargs)
        # overwrite the stride size
        if stride != self.model.patch_embed.proj.stride[0]:
            self.model.patch_embed.proj.stride = [stride, stride]

            def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
                """Get grid (feature) size for given image size taking account of dynamic padding.
                NOTE: must be torchscript compatible so using fixed tuple indexing
                """
                return (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1, (
                    img_size[1] - self.patch_size[1]
                ) // self.proj.stride[1] + 1

            self.model.patch_embed.dynamic_feat_size = types.MethodType(
                dynamic_feat_size, self.model.patch_embed
            )

    @property
    def n_output_dims(self) -> int:
        return self.model.pos_embed.shape[-1]

    @property
    def num_blocks(self) -> int:
        return len(self.model.blocks)

    @property
    def last_layer_index(self) -> int:
        return self.num_blocks - 1

    def create_model(
        self, model_identifier: str, **kwargs
    ) -> Tuple[Union[VisionTransformer, Eva], transforms.Compose]:
        model = timm.create_model(
            model_identifier,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=self.dynamic_img_size,
            dynamic_img_pad=self.dynamic_img_pad,
            **kwargs,
        )
        # Different models have different data configurations
        # e.g., their training resolution, normalization, etc, are different
        data_config = timm.data.resolve_model_data_config(model=model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return model, transforms

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
        reshape: bool = True,
        return_prefix_tokens: bool = False,
        norm: bool = True,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Intermediate layer accessor inspired by DINO / DINOv2 interface.
        Args:
            x: Input tensor.
            n: Take last n blocks if int, all if None, select matching indices if sequence
            reshape: Whether to reshape the output.
        """
        return self.model.forward_intermediates(
            x,
            n,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            output_fmt="NCHW" if reshape else "NLC",
            intermediates_only=True,
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
