try:
    import tinycudann as tcnn
except:
    print("tinycudann not found")
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class NeuralFeatureField(nn.Module):
    """A neural field that maps 2D coordinates to features."""

    def __init__(
        self,
        feat_dim: int = 768,
        base_resolution: int = 16,
        max_resolution: int = 1024,
        n_levels: int = 10,
        n_features_per_level: int = 8,
        log2_hashmap_size: int = 20,
    ):
        super(NeuralFeatureField, self).__init__()

        self.neural_field = tcnn.Encoding(
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
        self.mlp = nn.Sequential(
            nn.Linear(self.neural_field.n_output_dims, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, feat_dim),
        )

    def forward(self, coords: Tensor):
        assert coords.max() <= 1 and coords.min() >= 0, "coordinates should be in [0, 1]"
        denoised_features = self.neural_field(coords.view(-1, 2))
        return self.mlp(denoised_features.view(list(coords.shape[:-1]) + [-1]))
