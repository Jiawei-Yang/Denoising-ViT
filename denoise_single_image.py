import argparse
import logging
import os
import time
from itertools import chain
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from tqdm import tqdm

import dataset
import DenoisingViT
import utils.logging
from utils import misc
from utils.visualization_tools import (
    get_cluster_map,
    get_pca_map,
    get_scale_map,
    get_similarity_map,
)


def get_args_parser():
    parser = argparse.ArgumentParser("Single Image Denoising", add_help=False)
    parser.add_argument(
        "--config", default="configs/single_img_denoising.yaml", type=str
    )
    parser.add_argument(
        "--img_path", default=None, type=str,
        help="path to the image to denoise, will overwrite the image list in the config file",
    )
    parser.add_argument(
        "--skip_saving", action="store_true", help="whether to skip saving the results"
    )
    parser.add_argument(
        "--output_root",
        default="./work_dirs/",
        type=str,
        help="path to save checkpoints and logs",
    )
    parser.add_argument(
        "--batch_size", default=256, type=int, help="batch size to extract features"
    )
    parser.add_argument(
        "--ratio",
        default=1.0,
        type=float,
        help="ratio of the depth of the vit layers to use",
    )
    parser.add_argument("--project", default="denosing-vit", type=str)
    parser.add_argument("--run_name", default="debug", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def visualize_samples(
    denoiser: DenoisingViT.SingleImageDenoiser,
    neural_field: DenoisingViT.NeuralField,
    raw_features: Tensor,
    coord: Tensor,
    patch_images: Tensor,
    device: torch.device = torch.device("cuda"),
    denormalizer: transforms.Normalize = None,
):
    pca_samples = []
    for i in range(len(raw_features)):
        data_dict = {
            "transformed_view": patch_images[i : i + 1].to(device),
            "pixel_coords": coord[i : i + 1].to(device),
        }
        img = data_dict["transformed_view"]
        with torch.no_grad():
            output = denoiser.forward(
                raw_vit_outputs=raw_features[i : i + 1],
                pixel_coords=data_dict["pixel_coords"],
                neural_field=neural_field,
                return_visualization=True,
            )
        # gt noisy features
        gt_raw_features = output["raw_vit_outputs"]
        # compute the similarity of the central patch to the rest of the image
        gt_similarity_color = get_similarity_map(gt_raw_features, img.shape[-2:])
        gt_feature_norm_color = get_scale_map(
            torch.norm(gt_raw_features, dim=-1, keepdim=True), img.shape[-2:]
        )
        gt_cluster_map = get_cluster_map(
            gt_raw_features,
            img.shape[-2:],
            num_clusters=5,
        )

        gt_noisy_color = get_pca_map(
            gt_raw_features,
            img.shape[-2:],
        )

        # noise features
        noise_features = output["noise_features"]
        noise_color = get_pca_map(noise_features, img.shape[-2:])
        noise_norm = get_scale_map(
            torch.norm(noise_features, dim=-1, keepdim=True), img.shape[-2:]
        )
        # denoised_semantic_features from inr
        pred_denoised_cluster_map = get_cluster_map(
            output["denoised_semantic_features"],
            img.shape[-2:],
            num_clusters=5,
        )
        pred_denoised_color = get_pca_map(
            output["denoised_semantic_features"],
            img.shape[-2:],
        )
        pred_denoised_similarity_color = get_similarity_map(
            output["denoised_semantic_features"], img.shape[-2:]
        )
        pred_denoised_norm_color = get_scale_map(
            torch.norm(output["denoised_semantic_features"], dim=-1, keepdim=True),
            img.shape[-2:],
        )

        # real residual features
        gt_residual_features = (
            output["raw_vit_outputs"]
            - output["denoised_semantic_features"]
            - noise_features
        )
        gt_residual_norm = get_scale_map(
            torch.norm(gt_residual_features, dim=-1, keepdim=True),
            img.shape[-2:],
        )

        # undo standardization
        img = denormalizer(img)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()

        pca_sample = [
            img,
            gt_noisy_color,
            gt_cluster_map,
            gt_feature_norm_color,
            gt_similarity_color,
            pred_denoised_color,
            pred_denoised_cluster_map,
            pred_denoised_norm_color,
            pred_denoised_similarity_color,
            noise_color,
            noise_norm,
            gt_residual_norm,
        ]
        if "pred_residual" in output:
            # the norm of residual features
            pred_residual_norm = get_scale_map(
                torch.norm(output["pred_residual"], dim=-1, keepdim=True),
                img.shape[:2],
            )
            # combine noise features and residual features
            noise_features_and_residual = output["noise_features_and_residual"]
            noise_features_and_residual_color = get_pca_map(
                noise_features_and_residual, img.shape[:2]
            )
            # the norm of full residual features
            pca_sample.append(pred_residual_norm)
            pca_sample.append(noise_features_and_residual_color)
        pca_sample = np.concatenate(pca_sample, axis=1)
        pca_samples.append(pca_sample)
    pca_samples = np.concatenate(pca_samples, axis=0)
    pca_samples = (pca_samples * 255).astype(np.uint8)
    pca_samples = Image.fromarray(pca_samples)  # .resize(
    # (pca_samples.shape[1] // 2, pca_samples.shape[0] // 2)
    # )
    # the last sample is the full image
    return pca_samples, output["denoised_semantic_features"].detach().cpu().numpy()


def build_optimizer_and_lr_scheduler(
    cfg: OmegaConf,
    denoiser: DenoisingViT.SingleImageDenoiser,
    neural_field: DenoisingViT.NeuralField,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.Adam(
        chain(denoiser.parameters(), neural_field.parameters()),
        lr=cfg.optim.lr,
        eps=1e-15,
        weight_decay=cfg.optim.weight_decay,
        betas=(0.9, 0.99),
    )
    # a very naive lr scheduler I copied from my another project
    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=cfg.optim.num_iters // 10,
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    cfg.optim.num_iters * 1 // 4,
                    cfg.optim.num_iters * 2 // 4,
                    cfg.optim.num_iters * 3 // 4,
                    cfg.optim.num_iters * 9 // 10,
                ],
                gamma=0.33,
            ),
        ]
    )
    return optimizer, lr_scheduler


def build_models(
    cfg: OmegaConf, device: torch.device = torch.device("cuda")
) -> Tuple[DenoisingViT.SingleImageDenoiser, DenoisingViT.NeuralField]:
    # ---- build a denoiser ----#
    denoiser = DenoisingViT.SingleImageDenoiser(
        noise_map_height=cfg.model.denoiser.noise_map_height,
        noise_map_width=cfg.model.denoiser.noise_map_width,
        feature_dim=cfg.model.vit.feature_dim,
        layer_index=cfg.model.denoiser.layer_index,
        enable_residual_predictor=cfg.model.denoiser.enable_residual_predictor,
    ).to(device)
    # ---- build a neural field ----#
    neural_field = DenoisingViT.NeuralField(
        base_resolution=cfg.model.neural_field.base_resolution,
        max_resolution=cfg.model.neural_field.max_resolution,
        n_levels=cfg.model.neural_field.n_levels,
        n_features_per_level=cfg.model.neural_field.n_features_per_level,
        log2_hashmap_size=cfg.model.neural_field.log2_hashmap_size,
        feat_dim=cfg.model.vit.feature_dim,
    ).to(device)
    return denoiser, neural_field


def make_patch_coordinates(height, width, start: int = -1, end: int = 1) -> Tensor:
    patch_y = torch.linspace(start, end, height)
    patch_x = torch.linspace(start, end, width)
    patch_y, patch_x = torch.meshgrid(patch_y, patch_x, indexing="ij")
    patch_coordinates = torch.stack([patch_x, patch_y], dim=-1)
    return patch_coordinates


def denoise_an_image(
    all_raw_features: Tensor,
    all_pixel_coords: Tensor,
    all_transformed_views: Tensor,
    device: torch.device,
    cfg: OmegaConf = None,
    img_pth: str = None,
    denormalizer: transforms.Normalize = None,
) -> None:
    denoiser, neural_field = build_models(cfg, device)
    optimizer, lr_scheduler = build_optimizer_and_lr_scheduler(
        cfg, denoiser, neural_field
    )
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    patch_coords = make_patch_coordinates(
        cfg.model.denoiser.noise_map_height, cfg.model.denoiser.noise_map_width
    ).to(device)
    batched_patch_coords = patch_coords.unsqueeze(0).repeat(
        all_raw_features.shape[0], 1, 1, 1
    )
    batched_patch_coords = batched_patch_coords.reshape(-1, 2)
    batched_raw_features = all_raw_features.reshape(-1, all_raw_features.shape[-1])
    batched_pixel_coordinates = all_pixel_coords.reshape(-1, 2)

    for step in range(cfg.optim.num_iters):
        denoiser.train()
        neural_field.train()
        if step > int(cfg.optim.freeze_shared_artifacts_after * cfg.optim.num_iters):
            denoiser.stop_shared_artifacts_grad()
            denoiser.start_residual_predictor()
        random_pixel_indices = np.random.randint(0, batched_raw_features.shape[0], 2048)
        raw_features = batched_raw_features[random_pixel_indices]
        patch_coords = batched_patch_coords[random_pixel_indices]
        pixel_coordinates = batched_pixel_coordinates[random_pixel_indices]

        output = denoiser(
            raw_features,
            pixel_coordinates,
            neural_field=neural_field,
            patch_coords=patch_coords,
            return_visualization=False,
        )

        loss = output["loss"]

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        optimizer.step()
        lr_scheduler.step()

        if step % cfg.logging.print_freq == 0 or step == cfg.optim.num_iters - 1:
            logger.info(
                f"Step {step}/{cfg.optim.num_iters - 1}: "
                f"Loss = {loss.item():.4f}, "
                f"Patch Loss = {output['patch_l2_loss'].item():.4f}, "
                f"CosSim Loss = {output['cosine_similarity_loss'].item():.4f}, "
                f"Residual Loss = {output['residual_loss'].item() if 'residual_loss' in output else 0:.4f}, "
                f"Residual Sparsity Loss = {output['residual_sparsity_loss'].item() if 'residual_sparsity_loss' in output else 0:.4f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.4f}"
            )

    vis_indices = np.random.randint(0, all_raw_features.shape[0], cfg.logging.num_vis_samples)
    vis_indices = np.concatenate([vis_indices, [-1]])
    train_pca_samples, pred_full_denoised_features = visualize_samples(
        denoiser=denoiser,
        neural_field=neural_field,
        raw_features=all_raw_features[vis_indices],
        coord=all_pixel_coords[vis_indices],
        patch_images=all_transformed_views[vis_indices],
        device=device,
        denormalizer=denormalizer,
    )
    os.makedirs(f"{cfg.log_dir}/visualization", exist_ok=True)
    img_name = os.path.basename(img_pth)
    dir_name = os.path.dirname(img_pth)
    train_pca_samples.save(f"{cfg.log_dir}/visualization/{img_name}")
    logger.info(f"Saved visualization to {cfg.log_dir}/visualization/{img_name}")
    # designed for VOC
    if not cfg.skip_saving:
        raw_feat_dir = f"{dir_name.replace('JPEGImages', 'raw_features')}/{cfg.model.vit.type}_s{cfg.model.vit.stride}/"
        denoised_feat_dir = f"{dir_name.replace('JPEGImages', 'denoised_features')}/{cfg.model.vit.type}_s{cfg.model.vit.stride}/"
        os.makedirs(raw_feat_dir, exist_ok=True)
        os.makedirs(denoised_feat_dir, exist_ok=True)
        np.save(
            f"{denoised_feat_dir}/{img_name.replace('.jpg', '.npy')}",
            pred_full_denoised_features,
        )
        np.save(
            f"{raw_feat_dir}/{img_name.replace('.jpg', '.npy')}",
            all_raw_features[-1].detach().cpu().numpy(),
        )
        logger.info(
           f"Saved denoised features to {denoised_feat_dir}/{img_name.replace('.jpg', '.npy')}"
        )
    
    del denoiser, neural_field, optimizer, lr_scheduler, grad_scaler
    torch.cuda.empty_cache()


def main(args: argparse.Namespace):
    # load config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    cfg.log_dir = log_dir
    cfg.skip_saving = args.skip_saving
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{log_dir}/visualization", exist_ok=True)
    # set up logging
    global logger
    utils.logging.setup_logging(
        output=log_dir,
        level=logging.INFO,
        time_string=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
    )
    logger = logging.getLogger()
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    # set random seeds
    misc.fix_random_seeds(cfg.optim.seed)
    cudnn.benchmark = True
    logger.info("Fix random seeds to {}".format(cfg.optim.seed))

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---- build models ----#
    logger.info("=> creating vit")
    vit = DenoisingViT.ViTWrapper(model_type=cfg.model.vit.type, stride=cfg.model.vit.stride)
    vit = vit.to(device).eval()
    cfg.model.vit.feature_dim = vit.n_output_dims
    layer_index = int(vit.last_layer_index * args.ratio)
    cfg.model.denoiser.layer_index = layer_index
    # overwrite the config
    cfg.model.denoiser.noise_map_height = (
        int(cfg.data.input_size[0]) - vit.patch_size[0]
    ) // cfg.model.vit.stride + 1
    cfg.model.denoiser.noise_map_width = (
        int(cfg.data.input_size[1]) - vit.patch_size[0]
    ) // cfg.model.vit.stride + 1

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}")

    # extract the normalization from timm config
    normalizer = vit.transformation.transforms[-1]
    assert isinstance(normalizer, transforms.Normalize), "last transform must be norm"
    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(normalizer.mean, normalizer.std)],
        std=[1 / s for s in normalizer.std],
    )
    if args.img_path is not None:
        img_paths = [args.img_path]
    else:
        img_paths = open(cfg.data.image_list, "r").readlines()
        img_paths = [
            line.strip()
        for line in img_paths[
            cfg.data.start_idx : cfg.data.start_idx + cfg.data.num_imgs
        ]
    ]
    for idx, img_pth in tqdm(
        enumerate(img_paths), desc="Training", dynamic_ncols=True, total=len(img_paths)
    ):
        # Check to see if the image is already denoised
        # If it is, then skip it unless we want to overwrite it
        dir_name = os.path.dirname(img_pth)
        fpath = f"{dir_name.replace('JPEGImages', 'denoised_features')}/{cfg.model.vit.type}_s{cfg.model.vit.stride}/"
        fpath += f"{os.path.basename(img_pth).replace('.jpg', '.npy')}"
        if os.path.exists(fpath) and not cfg.data.overwrite:
            logger.info(f"Skipping {img_pth} because it already exists")
            continue
        train_dataset = dataset.SingleImageDataset(
            img_pth=img_pth,
            size=cfg.data.input_size,
            base_transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalizer,
                ]
            ),
            final_transform=dataset.RandomResizedCropFlip(
                cfg.data.input_size,
                antialias=True,
                horizontal_flip=True,
                scale=(0.1, 0.5),
                patch_size=vit.patch_size[0],
                stride=cfg.model.vit.stride,
            ),
            num_iters=cfg.optim.num_iters,
        )

        collect_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,  # single patch loader
            num_workers=cfg.optim.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        pos_h, pos_w = (
            cfg.model.denoiser.noise_map_height,
            cfg.model.denoiser.noise_map_width,
        )
        num_samples = cfg.data.num_patches + 1
        pixel_coords = torch.zeros(
            (num_samples, pos_h, pos_w, 2),
            dtype=torch.float32,
            device=device,
        )
        patch_images = torch.zeros(
            (num_samples, 3, cfg.data.input_size[0], cfg.data.input_size[1]),
            dtype=torch.float32,
            device=device,
        )
        raw_features = torch.zeros(
            (num_samples, pos_h, pos_w, vit.n_output_dims),
            dtype=torch.float32,
            device=device,
        )
        counter = 0
        for i, data_dict in enumerate(collect_data_loader):
            if counter >= cfg.data.num_patches:
                break
            with torch.no_grad():
                _raw_features = vit.get_intermediate_layers(
                    data_dict["transformed_view"].to(device),
                    n=[layer_index],
                    reshape=True,
                )[-1].permute(0, 2, 3, 1)
                slicer = slice(counter, counter + len(_raw_features))
                pixel_coords[slicer] = data_dict["pixel_coords"]
                patch_images[slicer] = data_dict["transformed_view"]
                raw_features[slicer] = _raw_features
                counter += len(_raw_features)
        # add the full image
        with torch.no_grad():
            _raw_features = vit.get_intermediate_layers(
                data_dict["full_image"][0:1].to(device),
                n=[layer_index],
                reshape=True,
            )[-1].permute(0, 2, 3, 1)
            pixel_coords[-1] = make_patch_coordinates(
                cfg.model.denoiser.noise_map_height,
                cfg.model.denoiser.noise_map_width,
                start=0,
                end=1,
            ).to(device)
            patch_images[-1] = data_dict["full_image"][0]
            raw_features[-1] = _raw_features[0]
        denoise_an_image(
            raw_features,
            pixel_coords,
            patch_images,
            device=device,
            cfg=cfg,
            img_pth=img_pth,
            denormalizer=denormalizer,
        )
        print(f"Finished {idx + 1}/{len(img_paths)}")
        del pixel_coords, patch_images, raw_features
        del train_dataset, collect_data_loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
