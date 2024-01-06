import logging

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger()


class NeuralField(nn.Module):
    def __init__(
        self,
        base_resolution: int = 16,
        max_resolution: int = 1024,
        n_levels: int = 10,
        n_features_per_level: int = 8,
        log2_hashmap_size: int = 20,
        feat_dim: int = 768,
        verbose: bool = False,
    ):
        super(NeuralField, self).__init__()
        self.ingp = tcnn.Encoding(
            n_input_dims=2,
            dtype=torch.float32,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": np.exp(
                    (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
                ),
                "interpolation": "linear",
            },
        )
        if verbose:
            num_params = sum(p.numel() for p in self.ingp.parameters())
            logger.info(f"num_params: {num_params / 1e6}M")

        self.mlp = nn.Sequential(
            nn.Linear(self.ingp.n_output_dims, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, feat_dim),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, coords: Tensor):
        assert coords.max() <= 1 and coords.min() >= 0
        pred_denoised_features = self.ingp(coords.view(-1, 2)).view(
            list(coords.shape[:-1]) + [-1]
        )
        return self.mlp(pred_denoised_features)
