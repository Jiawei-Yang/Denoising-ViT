from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .neural_feature_field import NeuralFeatureField


class SingleImageDenoiser(nn.Module):
    def __init__(
        self,
        noise_map_height: int = 37,
        noise_map_width: int = 37,
        feat_dim: int = 768,
        layer_index: int = 11,
        enable_residual_predictor: bool = True,
        disable_pe: bool = False,
    ):
        super().__init__()
        self.noise_map_h = noise_map_height
        self.noise_map_w = noise_map_width
        self.feat_dim = feat_dim
        self.layer_idx = layer_index
        # g: the input-independent artifact term shared by all views
        if disable_pe:
            self.shared_artifacts = nn.Parameter(
                torch.zeros(1, feat_dim, noise_map_height, noise_map_width),
                requires_grad=False,
            )
        else:
            self.shared_artifacts = nn.Parameter(
                torch.randn(1, feat_dim, noise_map_height, noise_map_width) * 0.02,
                requires_grad=True,
            )
        self.enable_residual_predictor = enable_residual_predictor
        if self.enable_residual_predictor:
            # h: the residual term that is co-dependent on the input image and the spatial location
            self.residual_predictor = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, feat_dim),
            )
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
        global_pixel_coords: Tensor,
        neural_field: NeuralFeatureField = None,
        shared_artifact_coords: Tensor = None,
        return_visualization: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass of the SingleImageDenoiser.

        Args:
            raw_vit_outputs (Tensor): Raw outputs from the Vision Transformer (ViT).
            global_pixel_coords (Tensor): Normalized pixel coordinates (range [0, 1])
                of patch centers in the original image (what we call global image).
            neural_field (NeuralField, optional): Maps 2D coordinates to clean semantic features.
            shared_artifact_coords (Tensor, optional): Normalized pixel coordinates (range [0, 1])
                of patch centers in the **current view**.
            return_visualization (bool, optional): Whether to return visualization data.

        Returns:
            Dict[str, Tensor]: Dictionary containing loss values and, optionally, visualization data.
        """
        if len(raw_vit_outputs.shape) != 2:
            # (H, W, C) -> (N, C)
            original_shape = raw_vit_outputs.shape
            raw_vit_outputs = raw_vit_outputs.reshape(-1, self.feat_dim)
            global_pixel_coords = global_pixel_coords.reshape(-1, 2)
            # get the image-independent noise artifact term
            shared_patterns = self.shared_artifacts.permute(0, 2, 3, 1).reshape(-1, self.feat_dim)
        else:
            assert shared_artifact_coords is not None, "shared_artifact_coords must be provided."
            original_shape = None
            # get the image-independent noise artifact term
            shared_patterns = F.grid_sample(
                self.shared_artifacts,
                shared_artifact_coords[None, None, ...],
                mode="bilinear",
                align_corners=True,
            )
            shared_patterns = shared_patterns.squeeze().permute(1, 0)
        # 1. retrieve the denoised features from the holistic neural representation
        denoised_feats = neural_field(global_pixel_coords)
        # 3. get image-dependent high-norm / high-frequency patterns
        if self.use_residual_predictor:
            pred_residual = self.residual_predictor(raw_vit_outputs)
        else:
            pred_residual = None

        # 4. add shared artifact and residual features to the predicted denoised features
        # a holistic noise model
        if self.use_residual_predictor:
            # raw_vit_outputs = clean features + image-independent patterns + image-dependent patterns
            pred_raw_vit_outputs = denoised_feats + shared_patterns + pred_residual.detach()
        else:
            # raw_vit_outputs = clean features + image-independent patterns
            pred_raw_vit_outputs = shared_patterns + denoised_feats

        # ---- training loss ----#
        # 1. patch l2 loss, this loss optimizes the implicit neural representation and the image-independent patterns
        patch_l2_loss = F.mse_loss(pred_raw_vit_outputs, raw_vit_outputs)
        cosine_similarity = F.cosine_similarity(pred_raw_vit_outputs, raw_vit_outputs, dim=-1)
        cosine_similarity_loss = 1 - cosine_similarity.mean()
        loss = patch_l2_loss + cosine_similarity_loss
        results = {
            "patch_l2_loss": patch_l2_loss,
            "loss": loss,
            "cosine_similarity_loss": cosine_similarity_loss,
        }
        if self.use_residual_predictor:
            # 2. residual loss, this loss optimizes the image-dependent patterns
            gt_residual = (raw_vit_outputs - denoised_feats - shared_patterns).detach()
            residual_loss = 0.1 * F.mse_loss(pred_residual, gt_residual)
            loss += residual_loss
            # 3. residual sparsity loss, this loss optimizes the sparsity of the image-dependent patterns
            residual_sparsity_loss = 0.02 * pred_residual.abs().mean()
            loss += residual_sparsity_loss
            results["residual_loss"] = residual_loss
            results["residual_sparsity_loss"] = residual_sparsity_loss

        if return_visualization:
            assert original_shape is not None, "original_shape must be provided."
            # noisy gt features
            results["raw_vit_outputs"] = raw_vit_outputs.detach().reshape(*original_shape[:-1], -1)
            # predicted noisy features
            results["pred_features"] = pred_raw_vit_outputs.detach().reshape(
                *original_shape[:-1], -1
            )
            # predicted denoised features
            results["denoised_feats"] = denoised_feats.detach().reshape(*original_shape[:-1], -1)
            # predicted shared artifact (image-independent)
            results["shared_patterns"] = shared_patterns.detach().reshape(*original_shape[:-1], -1)
            if self.use_residual_predictor:
                # predicted residual features (image-dependent)
                results["pred_residual"] = pred_residual.detach().reshape(*original_shape[:-1], -1)
                # combine image-independent and image-dependent patterns
                shared_patterns_and_residual = shared_patterns + pred_residual
                results[
                    "shared_patterns_and_residual"
                ] = shared_patterns_and_residual.detach().reshape(*original_shape[:-1], -1)
                # real denoised feature maps
                denoised_features = raw_vit_outputs - shared_patterns - pred_residual
            else:
                # real denoised feature maps
                denoised_features = raw_vit_outputs - shared_patterns
            results["denoised_features"] = denoised_features.detach().reshape(
                *original_shape[:-1], -1
            )

        return results
