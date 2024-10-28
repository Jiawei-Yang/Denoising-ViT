import argparse
import logging
import math
import os
import re
import sys
import time

import imageio
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

import dvt.dataset as dataset
import dvt.models as DVT
import dvt.utils.logging as logging_utils
from dvt.utils import misc
from dvt.utils.visualization.visualization_tools import visualize_online_denoised_samples


def get_args():
    parser = argparse.ArgumentParser("Train generalizable denoiser", add_help=False)
    # model
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch14_dinov2.lvd142m",
        choices=DVT.MODEL_LIST,
    )
    parser.add_argument("--num_blocks", type=int, default=1)
    # data
    parser.add_argument("--data_root", type=str, default="data/imagenet")
    parser.add_argument("--feat_root", type=str, default=None)
    parser.add_argument("--data_list_path", type=str, default=None)
    parser.add_argument("--input_size", type=int, default=518, nargs="+")
    parser.add_argument("--auto_stride", action="store_true", help="set stride size = patch size.")
    parser.add_argument("--stride_size", type=int, default=14, help="Stride size for the model.")
    parser.add_argument("--num_workers", default=8, type=int)
    # training
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU")
    parser.add_argument("--num_vis_samples", default=8, type=int)
    parser.add_argument("--num_iterations", default=40_000, type=int)
    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--blr", type=float, default=2.0e-04, help="abs_lr = blr * total_bs / 256")
    parser.add_argument("--min_lr", type=float, default=1.0e-06, help="for cosine scheduler")
    parser.add_argument("--warmup_iters", type=int, default=50_000, help="iterations to warmup LR")

    # logging
    parser.add_argument("--output_root", default="./work_dirs/", type=str)
    parser.add_argument("--save_freq", default=5000, type=int)
    parser.add_argument("--vis_freq", default=5000, type=int)
    parser.add_argument("--project", default="denosing-vit", type=str)
    parser.add_argument("--run_name", default="debug", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", "--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    args = parser.parse_args()

    if isinstance(args.input_size, int):
        args.input_size = (args.input_size, args.input_size)
    if args.auto_stride:
        args.stride_size = int(re.search(r"patch(14|16)", args.model).group(1))
        print(f"Auto set stride to {args.stride_size}")
    if (args.stride_size == 16 or args.stride_size == 8) and args.input_size[0] == 518:
        args.input_size = [512, 512]
        print(f"Set input size to {args.input_size}")
    assert args.input_size[0] % args.stride_size == 0, "height must be divisible by stride_size"
    assert args.input_size[1] % args.stride_size == 0, "width must be divisible by stride_size"
    return args


def main(args: argparse.Namespace):
    # set up environment variables
    misc.init_distributed_mode(args)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    if misc.is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{log_dir}/visualization", exist_ok=True)

    # set up logging
    global logger
    logging_utils.setup_logging(
        output=log_dir,
        level=logging.INFO,
        time_string=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
    )
    logger = logging.getLogger()
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    logger.info("Command line: " + " ".join(sys.argv))
    logger.info("Host: " + os.uname().nodename)
    logger.info("Time: " + time.asctime())
    # set random seeds
    misc.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    logger.info("Fix random seeds to {}".format(args.seed))

    # set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info("=> creating model '{}'".format(args.model))
    vit = DVT.PretrainedViTWrapper(model_identifier=args.model, stride=args.stride_size)
    pos_h = (args.input_size[0] - vit.patch_size) // args.stride_size + 1
    pos_w = (args.input_size[1] - vit.patch_size) // args.stride_size + 1

    args.feat_dim = vit.n_output_dims
    args.noise_map_height = pos_h
    args.noise_map_width = pos_w

    # extract the normalization layer from timm's transformation
    normalizer = vit.transformation.transforms[-1]
    assert isinstance(normalizer, transforms.Normalize), "last transform must be norm"
    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(normalizer.mean, normalizer.std)],
        std=[1 / s for s in normalizer.std],
    )
    del vit
    vit = None

    model = DVT.Denoiser(
        noise_map_height=pos_h,
        noise_map_width=pos_w,
        feat_dim=args.feat_dim,
        vit=vit,
        num_blocks=args.num_blocks,
    ).to(device)
    model_without_ddp = model
    logger.info("Model = %s" % str(model_without_ddp))
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module
    train_dataset = dataset.PairedListDataset(
        data_root=args.data_root,
        feat_root=args.feat_root,
        data_list=args.data_list_path,
        transform=transforms.Compose(
            [
                transforms.Resize(args.input_size, interpolation=Image.BICUBIC, antialias=True),
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
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    args.lr = args.blr * math.sqrt(args.batch_size * misc.get_world_size() / 256)
    logger.info(f"sqrt scaling learning rate; blr: {args.blr}, actual lr: {args.lr}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    lr = dict(
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=args.num_iterations,
        warmup_iters=int(args.num_iterations * 0.15),
        start_warmup_value=0,
    )
    lr_schedule = misc.CosineScheduler(**lr)
    logger.info("Optimizer = %s" % str(optimizer))
    logger.info("LR Scheduler = %s" % str(lr_schedule))

    model.train()
    end = time.time()
    metric_logger = logging_utils.MetricLogger(delimiter="  ")
    for step, data_dict in enumerate(
        metric_logger.log_every(
            data_loader,
            50,
            header="Train",
            n_iterations=args.num_iterations,
        )
    ):
        if step >= args.num_iterations:
            break
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(device, non_blocking=True)
        data_time = time.time() - end
        lr = lr_schedule[step]
        misc.apply_optim_scheduler(optimizer, lr)
        pred_denoised_feats = model(data_dict["original_feats"])
        l2_loss = F.mse_loss(pred_denoised_feats, data_dict["denoised_feats"])
        cosine_sim = F.cosine_similarity(pred_denoised_feats, data_dict["denoised_feats"], dim=-1)
        cosine_similarity_loss = 1 - cosine_sim.mean()
        loss = l2_loss + cosine_similarity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

        if misc.is_main_process():
            # save checkpoint
            if step % args.save_freq == 0 or step == args.num_iterations - 1:
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
                ckpt_path = f"{log_dir}/checkpoints/ckpt_{step:06d}.pth"
                torch.save(to_save, ckpt_path)
                abs_path = os.path.abspath(ckpt_path)
                latest_symlink = f"{log_dir}/checkpoints/latest.pth"
                try:
                    os.remove(latest_symlink)
                except FileNotFoundError:
                    pass
                logger.info(f"Removed checkpoint: {latest_symlink}")
                os.symlink(abs_path, latest_symlink)
                logger.info(f"Saved checkpoint to {ckpt_path}")
                logger.info(f"Created symlink: {latest_symlink} -> {ckpt_path}")

            if step % args.vis_freq == 0 or step == args.num_iterations - 1:
                pca_samples = visualize_online_denoised_samples(
                    data_dict=data_dict,
                    pred_denoised_feats=pred_denoised_feats.detach().float(),
                    denormalizer=denormalizer,
                    num_samples=min(args.num_vis_samples, args.batch_size),
                )
                imageio.imsave(f"{log_dir}/visualization/{step:05d}.png", pca_samples)
                logger.info(f"Saved visualization to {log_dir}/visualization/{step:05d}.png")

        end = time.time()


if __name__ == "__main__":
    args = get_args()
    main(args)
