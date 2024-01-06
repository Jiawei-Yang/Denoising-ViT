from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from DenoisingViT.neural_field import NeuralField


class SingleImageDenoiser(nn.Module):
    def __init__(
        self,
        noise_map_height: int = 37,
        noise_map_width: int = 37,
        feature_dim: int = 768,
        layer_index: int = 11,
        enable_residual_predictor: bool = False,
    ):
        super().__init__()
        self.noise_map_h = noise_map_height
        self.noise_map_w = noise_map_width
        self.feature_dim = feature_dim
        self.layer_idx = layer_index
        # Input-independent noise artifact term shared by all views
        self.shared_artifacts = nn.Parameter(
            torch.randn(1, feature_dim, noise_map_height, noise_map_width) * 0.05,
            requires_grad=True,
        )
        # the residual term that is co-dependent on the input image and the spatial location
        self.enable_residual_predictor = enable_residual_predictor
        if self.enable_residual_predictor:
            self.residual_predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, feature_dim),
            )
        # whether
        self.residual_predictor_start = False

    def start_residual_predictor(self):
        """Enables the training of the residual predictor."""
        self.residual_predictor_start = True

    @property
    def use_residual_predictor(self):
        """Checks if the residual predictor should be used."""
        return self.enable_residual_predictor and self.residual_predictor_start

    def stop_shared_artifacts_grad(self):
        """Stops gradient updates for the shared artifacts."""
        self.shared_artifacts.requires_grad = False

    def forward(
        self,
        raw_vit_outputs: Tensor,
        pixel_coords: Tensor,
        neural_field: NeuralField = None,
        patch_coords: Tensor = None,
        return_visualization: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass of the SingleImageDenoiser.

        Args:

        Returns:
            Dict[str, Tensor]: A dictionary of loss values and visualization data.
        """
        if len(raw_vit_outputs.shape) != 2:
            # (h, w, c) -> (N, C)
            original_shape = raw_vit_outputs.shape
            raw_vit_outputs = raw_vit_outputs.reshape(-1, self.feature_dim)
            pixel_coords = pixel_coords.reshape(-1, 2)
            # get the image-independent noise artifact term
            noise_features = self.shared_artifacts.permute(0, 2, 3, 1).reshape(
                -1, self.feature_dim
            )
        else:
            assert patch_coords is not None, "patch_coords must be provided."
            original_shape = None
            # get the image-independent noise artifact term
            noise_features = (
                F.grid_sample(
                    self.shared_artifacts,
                    patch_coords.reshape(1, 1, -1, 2),
                    align_corners=True,
                    mode="bilinear",
                )
                .permute(0, 2, 3, 1)
                .reshape(-1, self.feature_dim)
            )
        # 1. retrieve the denoised features from the holistic neural representation
        denoised_semantic_features = neural_field(pixel_coords)
        # 3. get image-dependent high-norm / high-frequency patterns
        if self.use_residual_predictor:
            pred_residual = self.residual_predictor(raw_vit_outputs)
        else:
            pred_residual = None

        # 4. add noise features and residual features to the predicted denoised features
        # a holistic noise model
        if self.use_residual_predictor:
            # raw_vit_outputs = clean features + image-independent patterns + image-dependent patterns
            pred_raw_vit_outputs = (
                denoised_semantic_features + noise_features + pred_residual.detach()
            )
        else:
            # raw_vit_outputs = clean features + image-independent patterns
            pred_raw_vit_outputs = noise_features + denoised_semantic_features

        # ---- training loss ----#
        # 1. patch l2 loss, this loss optimizes the implicit neural representation and the image-independent patterns
        patch_l2_loss = F.mse_loss(pred_raw_vit_outputs, raw_vit_outputs)
        cosine_similarity_loss = (
            1
            - F.cosine_similarity(
                pred_raw_vit_outputs,
                raw_vit_outputs,
                dim=-1,
            ).mean()
        )
        loss = patch_l2_loss + cosine_similarity_loss
        results = {
            "patch_l2_loss": patch_l2_loss,
            "loss": loss,
            "cosine_similarity_loss": cosine_similarity_loss,
        }
        if self.use_residual_predictor:
            # 2. residual loss, this loss optimizes the image-dependent patterns
            residual_loss = 0.1 * F.mse_loss(
                pred_residual,
                (
                    raw_vit_outputs - denoised_semantic_features - noise_features
                ).detach(),
            )
            loss += residual_loss
            # 3. residual sparsity loss, this loss optimizes the sparsity of the image-dependent patterns
            residual_sparsity_loss = 0.02 * pred_residual.abs().mean()
            loss += residual_sparsity_loss
            results["residual_loss"] = residual_loss
            results["residual_sparsity_loss"] = residual_sparsity_loss

        if return_visualization:
            assert original_shape is not None, "original_shape must be provided."
            # noisy gt features
            results["raw_vit_outputs"] = raw_vit_outputs.detach().reshape(
                *original_shape[:-1], -1
            )
            # predicted noisy features
            results["pred_features"] = pred_raw_vit_outputs.detach().reshape(
                *original_shape[:-1], -1
            )
            # predicted denoised features
            results[
                "denoised_semantic_features"
            ] = denoised_semantic_features.detach().reshape(*original_shape[:-1], -1)
            # predicted noise features (image-independent)
            results["noise_features"] = noise_features.detach().reshape(
                *original_shape[:-1], -1
            )
            if self.use_residual_predictor:
                # predicted residual features (image-dependent)
                results["pred_residual"] = pred_residual.detach().reshape(
                    *original_shape[:-1], -1
                )
                # combine image-independent and image-dependent patterns
                noise_features_and_residual = noise_features + pred_residual
                results[
                    "noise_features_and_residual"
                ] = noise_features_and_residual.detach().reshape(
                    *original_shape[:-1], -1
                )
                # real denoised feature maps
                denoised_features = raw_vit_outputs - noise_features - pred_residual
            else:
                # real denoised feature maps
                denoised_features = raw_vit_outputs - noise_features
            results["denoised_features"] = denoised_features.detach().reshape(
                *original_shape[:-1], -1
            )

        return results
