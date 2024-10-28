from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Block, Mlp
from torch import Tensor

from .vit_wrapper import PretrainedViTWrapper


class Denoiser(nn.Module):
    def __init__(
        self,
        noise_map_height: int = 37,
        noise_map_width: int = 37,
        feat_dim: int = 768,
        vit: PretrainedViTWrapper = None,
        enable_pe: bool = True,
        num_blocks: int = 1,
    ):
        super().__init__()
        self.vit = vit
        self.denoiser = Block(
            dim=feat_dim,
            num_heads=feat_dim // 64,
            mlp_ratio=4,
            qkv_bias=True,
            qk_norm=False,
            init_values=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            mlp_layer=Mlp,
        )
        if num_blocks > 1:
            self.denoiser = nn.Sequential(
                *[
                    Block(
                        dim=feat_dim,
                        num_heads=feat_dim // 64,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_norm=False,
                        init_values=None,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        act_layer=nn.GELU,
                        mlp_layer=Mlp,
                    )
                    for _ in range(num_blocks)
                ]
            )

        self.pos_embed = None
        if enable_pe:
            seq_len = noise_map_height * noise_map_width
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, feat_dim) * 0.02)
        if self.vit is not None:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(
        self,
        x,
        return_dict=False,
        return_channel_first=False,
        return_class_token=False,
        norm=True,
    ):
        class_tokens = None
        if self.vit is not None:
            with torch.no_grad():
                # (B, C, H, W)
                vit_outputs = self.vit.get_intermediate_layers(
                    x,
                    n=[self.vit.last_layer_index],
                    return_prefix_tokens=return_class_token,
                    norm=norm,
                )
                if return_class_token:
                    vit_outputs = vit_outputs[-1]
                    class_tokens = vit_outputs[1][:, 0]
                original_feats = vit_outputs[0].permute(0, 2, 3, 1)
                x = original_feats
        else:
            original_feats = x.clone()
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed(self.pos_embed, (h, w), num_prefix_tokens=0)
        x = self.denoiser(x)
        x = x.reshape(b, h, w, c)
        if return_channel_first:
            x = x.permute(0, 3, 1, 2)
        if return_dict:
            return {
                "denoised_feats": x,
                "original_feats": original_feats.detach(),
                "class_tokens": class_tokens.detach() if class_tokens is not None else None,
            }
        if return_class_token:
            assert class_tokens is not None
            return x, class_tokens
        return x
