import builtins
import datetime
import logging
import math
import os
import random
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import inf

logger = logging.getLogger()


def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if is_enabled() else 0


def get_world_size():
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    return get_global_rank() == 0


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(args.rank, args.dist_url, args.gpu),
        flush=True,
    )
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def remove_on_master(files):
    if is_main_process():
        for file in files:
            file.unlink()


def save_denoiser(
    model,
    remove_vit: bool = True,
    checkpoint_path: str = "checkpoint.pth",
):
    state_dict = model.state_dict()
    if remove_vit:
        for k, v in list(state_dict.items()):
            if "vit." in k:
                del state_dict[k]
    to_save = {
        "model": state_dict,
    }
    save_on_master(to_save, checkpoint_path)
    logger.info("Saved denoiser checkpoint to %s" % checkpoint_path)


def save_inr(
    model,
    checkpoint_path: str = "checkpoint.pth",
):
    state_dict = model.state_dict()
    to_save = {
        "model": state_dict,
    }
    save_on_master(to_save, checkpoint_path)
    logger.info("Saved inr checkpoint to %s" % checkpoint_path)


def save_model(
    args: Namespace,
    cfg: OmegaConf,
    epoch: int,
    model_without_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_root: str,
    max_keep_ckpts: int = 3,
) -> None:
    checkpoint_paths = [f"{checkpoint_root}/checkpoint_{epoch:03d}.pth"]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
            "cfg": OmegaConf.to_container(cfg) if cfg is not None else None,
        }
        save_on_master(to_save, checkpoint_path)

    # Find all checkpoint files and sort them by their epoch number
    checkpoint_files = sorted(
        Path(checkpoint_root).glob("checkpoint_*.pth"),
        key=lambda x: int(x.name.split("_")[1].split(".")[0]),
        reverse=True,
    )

    # Keep only the latest max_keep_ckpts checkpoints or those whose epoch number is divisible by 10
    to_keep, to_delete = [], []
    for checkpoint_file in checkpoint_files:
        epoch_number = int(checkpoint_file.name.split("-")[1].split(".")[0])
        if epoch_number % 10 == 0:
            continue
        elif len(to_keep) < max_keep_ckpts:
            to_keep.append(checkpoint_file)
        else:
            to_delete.append(checkpoint_file)
    remove_on_master(to_delete)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not (hasattr(args, "eval") and args.eval)
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def apply_optim_scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0
            + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
