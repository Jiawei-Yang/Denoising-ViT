import argparse
import copy
import json
import logging
import os
import os.path as osp
import re
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction
from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset as build_segmentation_dataset
from mmseg.utils import collect_env, get_device, get_root_logger, setup_multi_processes

import dvt.models as DVT
from evaluation.depth.apis import train_depther
from evaluation.depth.datasets import build_dataset as build_depth_dataset
from evaluation.eval_utils.misc import build_depther, build_segmentor

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def parse_args():
    parser = argparse.ArgumentParser("Linear Evaluation", add_help=False)
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--load-denoiser-from", help="the checkpoint file to load weights from")
    parser.add_argument(
        "--load-distilled-model-from", help="the checkpoint file to load weights from"
    )
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--backbone-type",
        default="vit_small_patch14_dinov2.lvd142m",
        help="timm model type",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="segmentation",
        choices=["segmentation", "depth"],
        help="task to train",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--diff_seed",
        action="store_true",
        help="Whether or not set different seeds for different ranks",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        "not be supported in version v0.22.0. Override some settings in the "
        "used config, the key-value pair in xxx=yyy format will be merged "
        "into config file. If the value to be overwritten is a list, it "
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        "marks are necessary and that no white space is allowed.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="resume from the latest checkpoint automatically.",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
        cfg.gpu_ids = range(1)
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

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
    logger.info("Arguments:\n")
    logger.info(json.dumps(vars(args), indent=2))
    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f"Set random seed to {seed}, " f"deterministic: {args.deterministic}")
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(args.config)

    #############################################
    # Initialize the backbone we are evaluating #
    #############################################
    device = torch.device("cuda")
    args.stride = int(re.search(r"patch(\d+)", args.backbone_type).group(1))
    vit = DVT.PretrainedViTWrapper(model_identifier=args.backbone_type, stride=args.stride)
    vit = vit.to(device)
    if args.load_distilled_model_from is not None:
        ckpt = torch.load(args.load_distilled_model_from)
        try:
            msg = vit.load_state_dict(ckpt["model"])
        except:
            msg = vit.load_state_dict(ckpt["denoiser"])
        logger.info(f"Missing keys: {msg.missing_keys}")
        logger.info(f"Unexpected keys: {msg.unexpected_keys}")
        logger.info(f"{msg}")
        logger.info(f"Loaded distilled model from {args.load_distilled_model_from}")

    if args.load_denoiser_from is not None:
        if args.stride == 14:
            pos_h = 37
        elif args.stride == 16:
            pos_h = 32
        denoised_vit = DVT.Denoiser(
            noise_map_height=pos_h,
            noise_map_width=pos_h,
            feat_dim=vit.n_output_dims,
            vit=vit,
            num_blocks=args.num_blocks,
        )
        denoised_vit_ckpt = torch.load(args.load_denoiser_from)["denoiser"]
        msg = denoised_vit.load_state_dict(denoised_vit_ckpt, strict=False)
        missing_keys = {k for k in msg.missing_keys if "vit.model" not in k}
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {msg.unexpected_keys}")
        denoised_vit.to(device)
        backbone_model = denoised_vit
    else:
        backbone_model = vit

    backbone_model.eval()
    logger.info(backbone_model)
    if args.task == "segmentation":
        model = build_segmentor(cfg, backbone_model=backbone_model)
    elif args.task == "depth":
        model = build_depther(cfg, backbone_model=backbone_model)

    if cfg.get("SyncBN", False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            "SyncBN is only supported with DDP. To be compatible with DP, "
            "we convert SyncBN to BN. Please use dist_train.sh which can "
            "avoid this error."
        )
        model = revert_sync_batchnorm(model)

    logger.info(model)

    if args.task == "segmentation":
        build_dataset = build_segmentation_dataset
        trainer = train_segmentor
    elif args.task == "depth":
        build_dataset = build_depth_dataset
        trainer = train_depther

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    trainer(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
