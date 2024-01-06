# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import logging
import os
import os.path as osp
import time

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction
from mmseg.apis import set_random_seed
from mmseg.utils import collect_env, get_root_logger

import DeViT
from evaluation.depth.apis import train_depther
from evaluation.depth.datasets import build_dataset
from evaluation.eval_utils.misc import create_depther

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def parse_args():
    parser = argparse.ArgumentParser(description="Train a depthor")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--load-denoiser-from", help="the checkpoint file to load weights from"
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--backbone-type",
        default="vit_small_patch14_dinov2.lvd142m",
        help="timm model type",
    )
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--disable_pe", action="store_true", default=False)
    parser.add_argument("--denoiser_type", type=str, default="transformer")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    # if args.load_from is not None:
    #     cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg}")

    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, deterministic: " f"{args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)

    #############################################
    # Initialize the backbone we are evaluating #
    #############################################
    if args.backbone_type == "fbdino":
        BACKBONE_SIZE = "small"  # in ("small", "base", "large" or "giant")

        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        backbone_model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2", model=backbone_name
        )
        backbone_model.eval()
        backbone_model.cuda()
    elif args.backbone_type == "identity":
        backbone_model = torch.nn.Identity()
        backbone_model.cuda()
    else:
        device = torch.device("cuda")
        vit = DeViT.ViTWrapper(
            model_type=args.backbone_type,
            stride=args.stride,
        )
        vit = vit.to(device)
        logger.info(f"Loading backbone from {args.backbone_type}")

        # If we have a checkpoint for generalizable FreeViT, create a freevit model
        # and load the checkpoint into it.
        if args.load_denoiser_from is not None:
            if args.stride == 14:
                pos_h = 37
            elif args.stride == 16:
                pos_h = 32
            denoised_vit = DeViT.Denoiser(
                noise_map_height=pos_h,
                noise_map_width=pos_h,
                feature_dim=vit.n_output_dims,
                vit=vit,
                enable_pe=not args.disable_pe,
                denoiser_type=args.denoiser_type,
            )
            denoised_vit_ckpt = torch.load(args.load_denoiser_from)["denoiser"]
            msg = denoised_vit.load_state_dict(denoised_vit_ckpt, strict=False)
            logger.info(str(msg))
            for k in denoised_vit.state_dict().keys():
                if k in denoised_vit_ckpt:
                    logger.info(f"{k} loaded")
            denoised_vit.eval()
            denoised_vit.to(device)
            backbone_model = denoised_vit
        else:
            backbone_model = vit.model

        backbone_model.eval()

    model = create_depther(cfg, backbone_model=backbone_model)

    if cfg.get("SyncBN", False):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    # if cfg.checkpoint_config is not None:
    #     # save depth version, config file content and class names in
    #     # checkpoints as meta data
    #     cfg.checkpoint_config.meta = dict(
    #         config=cfg)
    # # passing checkpoint meta for saving best checkpoint
    # meta.update(cfg.checkpoint_config.meta)
    train_depther(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
