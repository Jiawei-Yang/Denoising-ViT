import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_kmeans import CosineSimilarity, KMeans


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


def get_pca_map(
    feature_map: torch.Tensor,
    img_size,
    interpolation="nearest",
    return_pca_stats=False,
    pca_stats=None,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1])
        )
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = F.interpolate(
        pca_color.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


def get_scale_map(
    scalar_map: torch.Tensor,
    img_size,
    interpolation="nearest",
):
    """
    scalar_map: (1, h, w, C) is the feature map of a single image.
    """
    if scalar_map.shape[0] != 1:
        scalar_map = scalar_map[None]
    scalar_map = (scalar_map - scalar_map.min()) / (scalar_map.max() - scalar_map.min() + 1e-6)
    scalar_map = F.interpolate(
        scalar_map.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    # cmap = plt.get_cmap("viridis")
    # scalar_map = cmap(scalar_map)[..., :3]
    # make it 3 channels
    scalar_map = torch.cat([scalar_map] * 3, dim=-1)
    scalar_map = scalar_map.cpu().numpy().squeeze(0)
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
    similarity_map = similarity_map_flat.view(H, W)

    # Normalize the similarity map to be in the range [0, 1] for visualization
    similarity_map = (similarity_map - similarity_map.min()) / (
        similarity_map.max() - similarity_map.min()
    )
    # we don't want the center patch to be the most similar
    similarity_map[H // 2, W // 2] = -1.0
    similarity_map = (
        F.interpolate(
            similarity_map.unsqueeze(0).unsqueeze(0),
            size=img_size,
            mode="bilinear",
        )
        .squeeze(0)
        .squeeze(0)
    )

    similarity_map_np = similarity_map.cpu().numpy()
    negative_mask = similarity_map_np < 0

    colormap = plt.get_cmap("turbo")

    # Apply the colormap directly to the normalized similarity map and multiply by 255 to get RGB values
    similarity_map_rgb = colormap(similarity_map_np)[..., :3]
    similarity_map_rgb[negative_mask] = [1.0, 0.0, 0.0]
    return similarity_map_rgb


def get_cluster_map(
    feature_map: torch.Tensor,
    img_size,
    num_clusters=10,
) -> torch.Tensor:
    kmeans = KMeans(n_clusters=num_clusters, distance=CosineSimilarity, verbose=False)
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    labels = kmeans.fit_predict(feature_map.reshape(1, -1, feature_map.shape[-1])).float()
    labels = (
        F.interpolate(labels.reshape(1, *feature_map.shape[:-1]), size=img_size, mode="nearest")
        .squeeze()
        .cpu()
        .numpy()
    ).astype(int)
    cmap = plt.get_cmap("rainbow", num_clusters)
    cluster_map = cmap(labels)[..., :3]
    return cluster_map.reshape(img_size[0], img_size[1], 3)
