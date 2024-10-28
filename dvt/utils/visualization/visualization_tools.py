import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_kmeans import CosineSimilarity, KMeans

import dvt.models as DVT

from .annotation import add_label, draw_label
from .layout import add_border, hcat, vcat


def get_robust_pca(features: Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(feat_map, img_size, interp="nearest", return_pca_stats=False, pca_stats=None):
    # make it B1, h, w, C)
    feat_map = feat_map[None] if feat_map.shape[0] != 1 else feat_map
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(feat_map.reshape(-1, feat_map.shape[-1]))
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feat_map @ reduct_mat
    pca_color = ((pca_color - color_min) / (color_max - color_min)).clamp(0, 1)
    pca_color = F.interpolate(pca_color.permute(0, 3, 1, 2), size=img_size, mode=interp)
    pca_color = pca_color.permute(0, 2, 3, 1).cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


def get_scale_map(scalar_map, img_size, interp="nearest"):
    """
    scalar_map: (1, h, w, C) is the feature map of a single image.
    """
    scalar_map = torch.norm(scalar_map, dim=-1, keepdim=True)
    if scalar_map.shape[0] != 1:
        scalar_map = scalar_map[None]
    scalar_map = (scalar_map - scalar_map.min()) / (scalar_map.max() - scalar_map.min() + 1e-6)
    scalar_map = F.interpolate(scalar_map.permute(0, 3, 1, 2), size=img_size, mode=interp)
    scalar_map = scalar_map.permute(0, 2, 3, 1).squeeze(-1)
    cmap = plt.get_cmap("inferno")
    scalar_map = cmap(scalar_map.float().cpu().numpy().squeeze(0))[..., :3]
    return scalar_map


def get_similarity_map(features: Tensor, img_size=(224, 224)):
    """
    compute the similarity map of the central patch to the rest of the image
    """
    assert len(features.shape) == 4, "features should be (1, C, H, W)"
    H, W, C = features.shape[1:]
    center_patch_feature = features[0, H // 2, W // 2, :]
    center_patch_feature_normalized = center_patch_feature / center_patch_feature.norm()
    center_patch_feature_normalized = center_patch_feature_normalized.unsqueeze(1)
    # Reshape and normalize the entire feature tensor
    features_flat = features.view(-1, C)
    features_normalized = features_flat / features_flat.norm(dim=1, keepdim=True)

    similarity_map_flat = features_normalized @ center_patch_feature_normalized
    # Reshape the flat similarity map back to the spatial dimensions (H, W)
    similarity_map = similarity_map_flat.view(1, 1, H, W)
    sim_min, sim_max = similarity_map.min(), similarity_map.max()
    similarity_map = (similarity_map - sim_min) / (sim_max - sim_min)

    # we don't want the center patch to be the most similar
    similarity_map[0, 0, H // 2, W // 2] = -1.0
    similarity_map = F.interpolate(similarity_map, size=img_size, mode="bilinear")
    similarity_map = similarity_map.squeeze(0).squeeze(0)

    similarity_map_np = similarity_map.float().cpu().numpy()
    negative_mask = similarity_map_np < 0

    colormap = plt.get_cmap("turbo")

    # Apply the colormap directly to the normalized similarity map and multiply by 255 to get RGB values
    similarity_map_rgb = colormap(similarity_map_np)[..., :3]
    similarity_map_rgb[negative_mask] = [1.0, 0.0, 0.0]
    return similarity_map_rgb


def get_cluster_map(feat_map, img_size, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, distance=CosineSimilarity, verbose=False)
    if feat_map.shape[0] != 1:
        feat_map = feat_map[None]  # make it (1, h, w, C)
    labels = kmeans.fit_predict(feat_map.reshape(1, -1, feat_map.shape[-1])).float()
    labels = F.interpolate(labels.reshape(1, *feat_map.shape[:-1]), size=img_size, mode="nearest")
    labels = labels.squeeze().cpu().numpy().astype(int)
    cmap = plt.get_cmap("rainbow", num_clusters)
    cluster_map = cmap(labels)[..., :3]
    return cluster_map.reshape(img_size[0], img_size[1], 3)


def visualize_offline_denoised_samples(
    denoiser: DVT.SingleImageDenoiser,
    neural_field: DVT.NeuralFeatureField,
    raw_features: torch.Tensor,
    coord: torch.Tensor,
    patch_images: torch.Tensor,
    device: torch.device = torch.device("cuda"),
    denormalizer=None,
    dtype=torch.float32,
):
    pca_samples = []
    for i in range(len(raw_features)):
        data_dict = {
            "transformed_view": patch_images[i : i + 1].to(device),
            "pixel_coords": coord[i : i + 1].to(device),
        }
        img = data_dict["transformed_view"]
        hw = img.shape[-2:]
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            output = denoiser.forward(
                raw_vit_outputs=raw_features[i : i + 1],
                global_pixel_coords=data_dict["pixel_coords"],
                neural_field=neural_field,
                return_visualization=True,
            )
        # gt noisy features
        gt_raw_features = output["raw_vit_outputs"].float()
        shared_patterns = output["shared_patterns"].float()
        denoised_feats = output["denoised_feats"].float()

        # compute the similarity of the central patch to the rest of the image
        gt_feature_pca = get_pca_map(gt_raw_features, hw)
        gt_cluster_map = get_cluster_map(gt_raw_features, hw, num_clusters=5)
        gt_norm_map = get_scale_map(gt_raw_features, hw)
        gt_similarity_map = get_similarity_map(gt_raw_features, hw)

        # shared artifact: G in the paper
        shared_artifact_pca = get_pca_map(shared_patterns, hw)
        # noise_norm = get_scale_map(shared_patterns, hw)

        # denoised_feats from neural fields: F in the paper
        denoised_pca = get_pca_map(denoised_feats, hw)
        denoised_cluster_map = get_cluster_map(denoised_feats, hw, num_clusters=5)
        denoised_norm_map = get_scale_map(denoised_feats, hw)
        denoised_similarity_map = get_similarity_map(denoised_feats, hw)

        # real residual features
        # gt_residual_features = gt_raw_features - denoised_feats - shared_patterns
        # gt_residual_norm = get_scale_map(gt_residual_features, hw)

        # undo standardization
        img = denormalizer(img)
        img = img.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

        pca_sample = [
            img,
            gt_feature_pca,
            gt_cluster_map,
            gt_norm_map,
            gt_similarity_map,
            denoised_pca,
            denoised_cluster_map,
            denoised_norm_map,
            denoised_similarity_map,
            shared_artifact_pca,
            # noise_norm,
            # gt_residual_norm,
        ]
        if "pred_residual" in output:  # h in the paper
            # the norm of residual features
            pred_residual_norm = get_scale_map(output["pred_residual"], hw)
            # combine shared artifact and residual features
            shared_patterns_and_residual = output["shared_patterns_and_residual"].float()
            shared_patterns_and_residual_color = get_pca_map(shared_patterns_and_residual, hw)
            # the norm of full residual features
            pca_sample.append(pred_residual_norm)
            pca_sample.append(shared_patterns_and_residual_color)
        pca_sample = [torch.tensor(sample).permute(2, 0, 1) for sample in pca_sample]
        if i == 0:
            pca_sample = hcat(
                add_label(pca_sample[0], "Input Image", font_size=58),
                add_label(pca_sample[1], "Original Feature", font_size=58),
                add_label(pca_sample[2], "Original Cluster", font_size=58),
                add_label(pca_sample[3], "Original Norm", font_size=58),
                add_label(pca_sample[4], "Original Sim", font_size=58),
                add_label(pca_sample[5], "Denoised Feat (F)", font_size=58),
                add_label(pca_sample[6], "Denoised Cluster", font_size=58),
                add_label(pca_sample[7], "Denoised Norm", font_size=58),
                add_label(pca_sample[8], "Denoised Sim", font_size=58),
                add_label(pca_sample[9], "Shared Noise (G)", font_size=58),
                add_label(pca_sample[10], "Residual Norm (h)", font_size=58),
                add_label(pca_sample[11], "Composited (G+h)", font_size=58),
                gap=12,
            )
        else:
            pca_sample = hcat(*pca_sample, gap=12)
        pca_samples.append(pca_sample)
    pca_samples = add_border(vcat(*pca_samples))
    pca_samples = pca_samples.permute(1, 2, 0).cpu().numpy()
    pca_samples = (pca_samples * 255).astype(np.uint8)
    return pca_samples, output["denoised_feats"].detach().float().cpu().numpy()


def visualize_online_denoised_samples(
    data_dict: dict,
    pred_denoised_feats: torch.Tensor,
    denormalizer=None,
    num_samples: int = 5,
):
    hw = data_dict["image"].shape[-2:]
    pca_samples = []
    for i in range(num_samples):
        image = denormalizer(data_dict["image"][i].cpu()).permute(1, 2, 0).numpy()
        original_feats = data_dict["original_feats"][i].float()
        gt_denoised_feats = data_dict["denoised_feats"][i].float()
        original_pca = get_pca_map(original_feats, hw)
        original_norm = get_scale_map(original_feats, hw)
        gt_denoised_color, pca_stats = get_pca_map(gt_denoised_feats, hw, return_pca_stats=True)
        gt_denoised_norm_color = get_scale_map(gt_denoised_feats, hw)
        pred_denoised_color = get_pca_map(pred_denoised_feats[i], hw, pca_stats=pca_stats)
        pred_denoised_norm_color = get_scale_map(pred_denoised_feats[i], hw)
        pca_sample = [
            image,
            original_pca,
            original_norm,
            gt_denoised_color,
            gt_denoised_norm_color,
            pred_denoised_color,
            pred_denoised_norm_color,
        ]
        pca_sample = [torch.tensor(sample).permute(2, 0, 1) for sample in pca_sample]
        if i == 0:
            pca_sample = hcat(
                add_label(pca_sample[0], "Input Image", font_size=58),
                add_label(pca_sample[1], "Original Feature", font_size=58),
                add_label(pca_sample[2], "Original Norm", font_size=58),
                add_label(pca_sample[3], "GT Denoised", font_size=58),
                add_label(pca_sample[4], "GT Denoised Norm", font_size=58),
                add_label(pca_sample[5], "Pred Denoised", font_size=58),
                add_label(pca_sample[6], "Pred Deno. Norm", font_size=58),
                gap=12,
            )
        else:
            pca_sample = hcat(*pca_sample, gap=12)
        pca_samples.append(pca_sample)

    pca_samples = add_border(vcat(*pca_samples))
    pca_samples = pca_samples.permute(1, 2, 0).cpu().numpy()
    pca_samples = (pca_samples * 255).astype(np.uint8)
    return pca_samples
