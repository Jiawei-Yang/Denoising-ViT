# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from timm.layers import resample_abs_pos_embed


@BACKBONES.register_module()
class DinoVisionTransformer(BaseModule):
    """Vision Transformer."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # windows: (num_windows*B, window_size, window_size, C)
    # windows = windows.view(-1, window_size * window_size, C)  # windows: (num_windows*B, window_size*window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    # x = x.view(B, -1, x.shape[-1])
    return x


def attn_forward(self, x):
    B, H, W, C = x.shape
    N = H * W
    x = x.view(B, H * W, C)
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x.view(B, H, W, C)


# Vision Transformer forward function
def get_vit_forward_fn(window_size):
    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if window_size > 0:
            x, pad_hw = window_partition(x, window_size)

        x = self.attn(x)
        # Reverse window partition
        if window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path1(self.ls1(x))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    return forward


def vit_pos_embed(self, x):
    # import ipdb; ipdb.set_trace()
    num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
    if self.dynamic_img_size:
        B, H, W, C = x.shape
        pos_embed = resample_abs_pos_embed(
            self.pos_embed[:, num_prefix_tokens:, :],
            (H, W),
            num_prefix_tokens=0,
        )
        # x = x.view(B, -1, C)
    else:
        pos_embed = self.pos_embed[:, num_prefix_tokens:, :]

    pos_embed = pos_embed.view(1, H, W, C)

    x = x + pos_embed

    return self.pos_drop(x)
