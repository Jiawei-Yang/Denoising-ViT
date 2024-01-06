import argparse
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image
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
    parser = argparse.ArgumentParser("Train generalizable denoiser", add_help=False)
    parser.add_argument("--config", default="configs/prompt_tuning.yaml", type=str)
    parser.add_argument(
        "--output_root",
        default="./work_dirs/",
        help="path to save checkpoints and logs",
        type=str,
    )
    parser.add_argument("--project", default="debug", type=str)
    parser.add_argument("--run_name", default="debug", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", "--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    return parser


def visualize_samples(
    cfg: OmegaConf,
    step: int,
    data_dict: dict,
    pred_denoised_feats: torch.Tensor,
    num_samples: int = 5,
    denormalizer: transforms.Normalize = None,
):
    H, W = cfg.data.input_size
    pca_samples = []
    for i in range(num_samples):
        if denormalizer is not None:
            image = denormalizer(data_dict["image"][i].cpu()).permute(1, 2, 0).numpy()
        else:
            image = data_dict["image"][i].permute(1, 2, 0).cpu().numpy()
            image = image * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
        gt_noisy_color = get_pca_map(
            data_dict["raw_vit_feats"][i],
            (H, W),
        )
        gt_noisy_norm_color = get_scale_map(
            torch.norm(data_dict["raw_vit_feats"][i], dim=-1, keepdim=True),
            (H, W),
        )
        gt_cluster_map = get_cluster_map(
            data_dict["raw_vit_feats"][i],
            (H, W),
            num_clusters=5,
        )
        gt_denoised_color, pca_stats = get_pca_map(
            data_dict["denoised_feats"][i],
            (H, W),
            return_pca_stats=True,
        )
        gt_denoised_norm_color = get_scale_map(
            torch.norm(data_dict["denoised_feats"][i], dim=-1, keepdim=True),
            (H, W),
        )
        pred_denoised_color = get_pca_map(
            pred_denoised_feats[i],
            (H, W),
            pca_stats=pca_stats,
        )
        pred_denoised_norm_color = get_scale_map(
            torch.norm(pred_denoised_feats[i], dim=-1, keepdim=True),
            (H, W),
        )
        pred_denoised_cluster_map = get_cluster_map(
            pred_denoised_feats[i],
            (H, W),
            num_clusters=5,
        )
        pca_samples.append(
            np.concatenate(
                [
                    image,
                    gt_noisy_color,
                    gt_cluster_map,
                    gt_noisy_norm_color,
                    gt_denoised_color,
                    gt_denoised_norm_color,
                    pred_denoised_color,
                    pred_denoised_cluster_map,
                    pred_denoised_norm_color,
                ],
                axis=1,
            )
        )
    pca_samples = np.concatenate(pca_samples, axis=0)
    pca_samples = Image.fromarray((pca_samples * 255).astype(np.uint8))
    pca_samples = pca_samples.resize(
        (pca_samples.size[0] // 2, pca_samples.size[1] // 2),
        resample=Image.BICUBIC,
    )
    pca_samples.save(f"{cfg.log_dir}/visualization/{step:05d}.png")
    logger.info(f"Saved visualization to {cfg.log_dir}/visualization/{step:05d}.png")


def make_list(cfg: OmegaConf):
    img_list = cfg.data.image_list
    with open(img_list, "r") as f:
        img_list = f.readlines()
        img_list = [line.strip() for line in img_list]
    data_list = []
    if cfg.data.num_max_samples > 0:
        img_list = img_list[: cfg.data.num_max_samples]
    for img_pth in tqdm(img_list, desc="making data list"):
        denoised_feat_pth = img_pth.replace(
            "JPEGImages",
            f"denoised_features/{cfg.model.vit.type}_s{cfg.model.vit.stride}",
        ).replace(".jpg", ".npy")
        raw_feat_pth = img_pth.replace(
            "JPEGImages",
            f"raw_features/{cfg.model.vit.type}_s{cfg.model.vit.stride}",
        ).replace(".jpg", ".npy")
        if os.path.exists(denoised_feat_pth) and os.path.exists(raw_feat_pth):
            data_list.append(f"{img_pth} {raw_feat_pth} {denoised_feat_pth}\n")
    logger.info(f"Found {len(data_list)} valid samples")
    cfg.data.data_list = cfg.data.image_list.replace(
        ".txt", f"_{cfg.model.vit.type}_s{cfg.model.vit.stride}.txt"
    )
    with open(cfg.data.data_list, "w") as f:
        f.writelines(data_list)
    return cfg


def main(args: argparse.Namespace):
    # set up environment variables
    misc.init_distributed_mode(args)
    # load config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    cfg.log_dir = log_dir
    if misc.is_main_process():
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
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    # build dataset
    logger.info("=> creating model '{}'".format(cfg.model.vit.type))
    vit = DenoisingViT.ViTWrapper(model_type=cfg.model.vit.type, stride=cfg.model.vit.stride)

    cfg.model.vit.feature_dim = vit.n_output_dims
    # overwrite the config
    cfg.model.denoiser.noise_map_height = (
        int(cfg.data.input_size[0]) - vit.patch_size[0]
    ) // cfg.model.vit.stride + 1
    cfg.model.denoiser.noise_map_width = (
        int(cfg.data.input_size[1]) - vit.patch_size[0]
    ) // cfg.model.vit.stride + 1

    # extract the normalization from timm config
    normalizer = vit.transformation.transforms[-1]
    assert isinstance(normalizer, transforms.Normalize), "last transform must be norm"
    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(normalizer.mean, normalizer.std)],
        std=[1 / s for s in normalizer.std],
    )

    if cfg.data.is_raw_feat_cached:
        logger.info("=> using cached feature map")
        del vit
        torch.cuda.empty_cache()
        vit = None
    else:
        logger.info("=> using ViT as backbone")
        vit = vit.to(device)

    model = DenoisingViT.Denoiser(
        noise_map_height=cfg.model.denoiser.noise_map_height,
        noise_map_width=cfg.model.denoiser.noise_map_width,
        feature_dim=cfg.model.vit.feature_dim,
        vit=vit,
        enable_pe=cfg.model.denoiser.enable_pe,
        denoiser_type=cfg.model.denoiser.denoiser_type,
    ).to(device)
    model_without_ddp = model

    logger.info("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
        model_without_ddp = model.module
    if cfg.data.data_list is None:
        cfg = make_list(cfg)
        # wait for all processes to synchronize
    train_dataset = dataset.PairedListDataset(
        data_list=cfg.data.data_list,
        transform=transforms.Compose(
            [
                transforms.Resize(
                    cfg.data.input_size, interpolation=Image.BICUBIC, antialias=True
                ),
                transforms.ToTensor(),
                normalizer,
            ]
        ),
    )
    logger.info(f"Dataset: {train_dataset}")
    logger.info(f"Dataset size: {len(train_dataset)}")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_global_rank()
        # Initialize the distributed infinite sampler
        sampler = dataset.DistributedInfiniteSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank
        )
        logger.info(f"Sampler: {sampler}")
    else:
        num_tasks = 1
        global_rank = 0
        sampler = dataset.InfiniteSampler(train_dataset)

    # Create the DataLoader
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.optim.batch_size,
        sampler=sampler,
        num_workers=cfg.optim.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    base_lr = cfg.optim.base_lr
    cfg.optim.lr = base_lr
    cfg.optim.lr *= math.sqrt(cfg.optim.batch_size * misc.get_world_size() / 256)

    logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    logger.info("LR = %f" % cfg.optim.lr)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
        weight_decay=cfg.optim.weight_decay,
    )
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim.min_lr,
        total_iters=cfg.optim.num_iters,
        warmup_iters=int(cfg.optim.num_iters * 0.15),
        start_warmup_value=0,
    )
    lr_schedule = misc.CosineScheduler(**lr)

    logger.info("Optimizer = %s" % str(optimizer))
    logger.info("LR Scheduler = %s" % str(lr_schedule))
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}")

    model.train()
    print_freq = cfg.logging.print_freq
    end = time.time()
    metric_logger = utils.logging.MetricLogger(delimiter="  ")
    for step, data_dict in enumerate(
        metric_logger.log_every(
            data_loader, print_freq, n_iterations=cfg.optim.num_iters
        )
    ):
        if step >= cfg.optim.num_iters:
            break
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(device, non_blocking=True)
        data_time = time.time() - end
        lr = lr_schedule[step]
        misc.apply_optim_scheduler(optimizer, lr)
        if cfg.data.is_raw_feat_cached:
            pred_denoised_feats = model(data_dict["raw_vit_feats"])
        else:
            assert model_without_ddp.vit is not None
            results = model(data_dict["image"], return_dict=True)
            data_dict["raw_vit_feats"] = results["raw_vit_feats"]
            pred_denoised_feats = results["pred_denoised_feats"]
        l2_loss = F.mse_loss(
            pred_denoised_feats,
            data_dict["denoised_feats"],
        )
        cosine_similarity_loss = (
            1
            - F.cosine_similarity(
                pred_denoised_feats,
                data_dict["denoised_feats"],
                dim=-1,
            ).mean()
        )
        loss = l2_loss + cosine_similarity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        if not torch.isfinite(loss):
            logger.error("Loss is {}, stopping training".format(loss))
            logger.error(pred_denoised_feats)
            sys.exit(1)

        torch.cuda.synchronize()
        iter_time = time.time() - end
        metric_logger.update(
            loss=loss.item(),
            l2_loss=l2_loss.item(),
            cosine_similarity_loss=cosine_similarity_loss.item(),
            data_time=data_time,
            iter_time=iter_time,
            lr=optimizer.param_groups[0]["lr"],
        )

        end = time.time()
        if misc.is_main_process():
            # save checkpoint
            if step % cfg.logging.save_freq == 0 or step == cfg.optim.num_iters - 1:
                model_state_dict = {}
                for k, v in list(model_without_ddp.state_dict().items()):
                    if "vit." in k:
                        continue
                    else:
                        model_state_dict[k] = v
                to_save = {
                    "denoiser": model_state_dict,
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                torch.save(
                    to_save,
                    f"{cfg.log_dir}/checkpoints/ckpt_{step:06d}.pth",
                )
                logger.info(
                    f"Saved checkpoint to {cfg.log_dir}/checkpoints/ckpt_{step:06d}.pth"
                )
            # visualization
            if step % cfg.logging.vis_freq == 0 or step == cfg.optim.num_iters - 1:
                visualize_samples(
                    cfg=cfg,
                    step=step,
                    data_dict=data_dict,
                    pred_denoised_feats=pred_denoised_feats.detach(),
                    num_samples=min(cfg.logging.num_vis_samples, cfg.optim.batch_size),
                    denormalizer=denormalizer,
                )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
