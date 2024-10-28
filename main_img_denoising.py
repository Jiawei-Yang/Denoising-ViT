import argparse
import datetime
import glob
import json
import os
import time
from itertools import chain

import imageio
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

import dvt.models as DVT
import dvt.utils.misc as misc
from dvt.dataset import RandomResizedCropFlip, SingleImageDataset
from dvt.utils.visualization.visualization_tools import visualize_offline_denoised_samples


def make_patch_coordinates(height, width, start=-1, end=1):
    patch_y, patch_x = torch.linspace(start, end, height), torch.linspace(start, end, width)
    patch_y, patch_x = torch.meshgrid(patch_y, patch_x, indexing="ij")
    patch_coordinates = torch.stack([patch_x, patch_y], dim=-1)
    return patch_coordinates


def denoise_an_image(
    args: argparse.Namespace,
    all_raw_features: torch.Tensor,
    all_pixel_coords: torch.Tensor,
    all_transformed_views: torch.Tensor,
    device: torch.device,
    denormalizer: transforms.Normalize = None,
    img_pth: str = None,
    should_save_vis: bool = False,
):
    # ---- build the models and optimizer ---- #
    denoiser = DVT.SingleImageDenoiser(
        noise_map_height=args.noise_map_height,
        noise_map_width=args.noise_map_width,
        feat_dim=args.feat_dim,
        layer_index=args.layer_index,
    ).to(device)
    # ---- build a neural field ----#
    neural_field = DVT.NeuralFeatureField(feat_dim=args.feat_dim, n_levels=args.n_levels)
    neural_field = neural_field.to(device)
    optimizer = torch.optim.Adam(
        chain(denoiser.parameters(), neural_field.parameters()),
        lr=args.lr,
        eps=1e-15,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    grad_scaler = torch.amp.GradScaler("cuda", 2**10)

    # ----- shared artifact (G) coordinates ----- #
    shared_artifact_coords = make_patch_coordinates(args.noise_map_height, args.noise_map_width)
    shared_artifact_coords = shared_artifact_coords.to(device)
    num_views = all_raw_features.shape[0]
    batched_shared_artifact_coords = shared_artifact_coords.unsqueeze(0).repeat(num_views, 1, 1, 1)
    batched_shared_artifact_coords = batched_shared_artifact_coords.reshape(-1, 2)

    batched_raw_features = all_raw_features.reshape(-1, all_raw_features.shape[-1])
    batched_pixel_coordinates = all_pixel_coords.reshape(-1, 2)

    for step in range(args.num_iters):
        denoiser.train()
        neural_field.train()
        if step > int(args.freeze_shared_artifacts_after * args.num_iters):
            denoiser.stop_shared_artifacts_grad()
            denoiser.start_residual_predictor()
        random_pixel_indices = np.random.randint(0, batched_raw_features.shape[0], args.pixel_bsz)
        raw_features = batched_raw_features[random_pixel_indices]
        shared_artifact_coords = batched_shared_artifact_coords[random_pixel_indices]
        pixel_coordinates = batched_pixel_coordinates[random_pixel_indices]
        misc.adjust_learning_rate(optimizer, step, args)
        with torch.autocast(device, dtype=args.dtype, enabled=args.dtype != torch.float32):
            output = denoiser(
                raw_vit_outputs=raw_features,
                global_pixel_coords=pixel_coordinates,
                neural_field=neural_field,
                shared_artifact_coords=shared_artifact_coords,
                return_visualization=False,
            )
            loss = output["loss"]
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        optimizer.step()

        if step % 1000 == 0 or step == args.num_iters - 1:
            print(
                f"Step {step}/{args.num_iters - 1}: "
                f"Loss = {loss.item():.4f}, "
                f"Patch Loss = {output['patch_l2_loss'].item():.4f}, "
                f"CosSim Loss = {output['cosine_similarity_loss'].item():.4f}, "
                f"Residual Loss = {output['residual_loss'].item() if 'residual_loss' in output else 0:.4f}, "
                f"Residual Sparsity Loss = {output['residual_sparsity_loss'].item() if 'residual_sparsity_loss' in output else 0:.4f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.4f}"
            )
    if should_save_vis:
        vis_indices = np.random.randint(0, num_views, args.num_vis_samples)
        vis_indices = np.concatenate([vis_indices, [-1]])
        train_pca_samples, pred_full_denoised_features = visualize_offline_denoised_samples(
            denoiser=denoiser,
            neural_field=neural_field,
            raw_features=all_raw_features[vis_indices],
            coord=all_pixel_coords[vis_indices],
            patch_images=all_transformed_views[vis_indices],
            device=device,
            denormalizer=denormalizer,
            dtype=args.dtype,
        )
        os.makedirs(f"{args.output_dir}/visualization", exist_ok=True)
        img_name = os.path.basename(img_pth)
        imageio.imsave(f"{args.output_dir}/visualization/{img_name}", train_pca_samples)
        print(f"Saved visualization to {args.output_dir}/visualization/{img_name}")
    else:
        pred_full_denoised_features = None

    if args.data_root is not None:
        if pred_full_denoised_features is None:
            with torch.no_grad():
                output = denoiser(
                    raw_vit_outputs=all_raw_features[-1:],
                    global_pixel_coords=all_pixel_coords[-1:],
                    neural_field=neural_field,
                    return_visualization=True,
                )
            pred_full_denoised_features = output["denoised_feats"].float().detach().cpu().numpy()
        raw_feat_dir = f"{args.save_root}/raw_features/{args.model}/"
        denoised_feat_dir = f"{args.save_root}/denoised_features/{args.model}/"
        img_extention = os.path.splitext(img_pth)[1]
        raw_feat_save_path = img_pth.replace(args.data_root, raw_feat_dir).replace(
            img_extention, ".npy"
        )
        denoised_feat_save_path = img_pth.replace(args.data_root, denoised_feat_dir).replace(
            img_extention, ".npy"
        )
        os.makedirs(os.path.dirname(raw_feat_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(denoised_feat_save_path), exist_ok=True)
        np.save(raw_feat_save_path, all_raw_features[-1].float().detach().cpu().numpy())
        np.save(denoised_feat_save_path, pred_full_denoised_features)
        print(
            f"Saved denoised features to {denoised_feat_save_path} and raw features to {raw_feat_save_path}"
        )

    del denoiser, neural_field, optimizer
    torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description="DVT Stage-1: Single Image Denoising")
    # Model and input arguments
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch14_dinov2.lvd142m",
        choices=DVT.MODEL_LIST,
        help="Identifier for the timm model.",
    )
    parser.add_argument("--input_size", type=int, default=518, nargs="+")
    parser.add_argument("--stride_size", type=int, default=14, help="Stride size for the model.")
    parser.add_argument(
        "--layer_depth_ratio",
        type=float,
        default=1.0,
        help="Ratio to determine which layer to denoise. 1.0 means denoising the last layer.",
    )

    # Data and saving arguments
    parser.add_argument("--img_path", type=str, default="demo/assets/demo/cat.jpg")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for the model.")
    parser.add_argument("--data_root", type=str, default=None, help="Path to the input image.")
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_imgs", type=int, default=100)

    # Denoising parameters
    parser.add_argument("--num_views", type=int, default=768, help="Num of views for denoising.")
    parser.add_argument("--num_iters", type=int, default=25000, help="Num of iters for denoising.")
    parser.add_argument("--warmup_iters", type=int, default=2500, help="Num of warmup iters.")
    parser.add_argument("--n_levels", type=int, default=16, help="Num of lvls for the instant-ngp")
    parser.add_argument(
        "--freeze_shared_artifacts_after",
        type=float,
        default=0.5,
        help="Freeze shared artifacts after this fraction of iterations.",
    )

    # Optimization parameters
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--min_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--extract_bsz", type=int, default=32, help="Batch size for feature extraction."
    )
    parser.add_argument("--pixel_bsz", type=int, default=2048, help="Batch size for pixels.")

    # Output and visualization
    parser.add_argument("--output_dir", type=str, default="./work_dirs/demo")
    parser.add_argument("--num_vis_samples", type=int, default=5, help="Num of views to visualize.")
    parser.add_argument("--vis_freq", type=int, default=100, help="Save visualization frequency.")

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()

    # Validate arguments
    assert os.path.isfile(args.img_path), f"Image not found: {args.img_path}"
    if isinstance(args.input_size, int):
        args.input_size = (args.input_size, args.input_size)
    assert args.input_size[0] % args.stride_size == 0, "height must be divisible by stride_size"
    assert args.input_size[1] % args.stride_size == 0, "width must be divisible by stride_size"

    return args


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    misc.fix_random_seeds(args.seed)
    print(f"Arguments:\n{json.dumps(vars(args), indent=4)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isfile(args.img_path):
        if args.img_path.endswith("txt"):
            with open(args.img_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, "**/*"), recursive=True)
    filenames = filenames[args.start_idx : args.start_idx + args.num_imgs]

    # prepare the model
    vit = DVT.PretrainedViTWrapper(model_identifier=args.model, stride=args.stride_size)
    vit = vit.to(device).eval()

    # update the arguments
    layer_index = int(args.layer_depth_ratio * vit.last_layer_index)
    pos_h = (args.input_size[0] - vit.patch_size) // args.stride_size + 1
    pos_w = (args.input_size[1] - vit.patch_size) // args.stride_size + 1
    args.layer_index = layer_index
    args.feat_dim = vit.n_output_dims
    args.noise_map_height = pos_h
    args.noise_map_width = pos_w

    # Extract the data normalization operation from timm's transformation
    normalizer = vit.transformation.transforms[-1]
    assert isinstance(normalizer, transforms.Normalize), "last transform must be norm"
    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(normalizer.mean, normalizer.std)],
        std=[1 / s for s in normalizer.std],
    )

    args.dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print("Using", args.dtype, "data type")

    # placeholder for the data
    num_samples = args.num_views + 1  # + 1 for the original image
    global_pixel_coords = torch.zeros(
        (num_samples, pos_h, pos_w, 2),
        dtype=args.dtype,
        device=device,
    )
    views = torch.zeros(
        (num_samples, 3, args.input_size[0], args.input_size[1]),
        dtype=args.dtype,
        device=device,
    )
    vit_features = torch.zeros(
        (num_samples, pos_h, pos_w, vit.n_output_dims),
        dtype=args.dtype,
        device=device,
    )

    dataset = SingleImageDataset(
        size=args.input_size,
        base_transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.ToTensor(),
                normalizer,
            ]
        ),
        final_transform=RandomResizedCropFlip(
            size=args.input_size,
            horizontal_flip=True,
            scale=(0.1, 0.5),
            patch_size=vit.patch_size,
            stride=args.stride_size,
        ),
        num_views=args.num_views,
    )

    num_processed_samples = 0
    autocast_ctx = torch.autocast("cuda", dtype=args.dtype, enabled=args.dtype != torch.float32)
    start_time = time.time()
    for idx, filename in enumerate(filenames):
        filename = filename.strip().split(" ")[0]
        if args.data_root is not None:
            filename = os.path.join(args.data_root, filename)
            if misc.check_if_file_exists(args, filename):
                print(f"Skipping {filename}")
                continue

        dataset.set_image(filename)
        collect_loader = torch.utils.data.DataLoader(dataset, args.extract_bsz, num_workers=8)
        torch.cuda.empty_cache()

        extract_start_time = time.time()
        pbar = tqdm(collect_loader, total=len(collect_loader), desc="Collecting features")
        for i, data in enumerate(pbar):
            with torch.no_grad(), autocast_ctx:
                batch_vit_features = vit.get_intermediate_layers(
                    data["transformed_view"].to(device),
                    n=[layer_index],
                    reshape=True,
                )[-1]
                # (B, C, H, W) -> (B, H, W, C)
                batch_vit_features = batch_vit_features.permute(0, 2, 3, 1)
                batch_pixel_coords = data["pixel_coords"].to(device)
                batch_views = data["transformed_view"].to(device)
                slicer = slice(i * args.extract_bsz, i * args.extract_bsz + batch_views.shape[0])
                global_pixel_coords[slicer] = batch_pixel_coords
                views[slicer] = batch_views
                vit_features[slicer] = batch_vit_features
        # add the original image to the list
        with torch.no_grad(), autocast_ctx:
            original_vit_features = vit.get_intermediate_layers(
                data["full_image"][:1].to(device), n=[layer_index], reshape=True
            )[-1]
            # (B, C, H, W) -> (B, H, W, C)
            original_vit_features = original_vit_features.permute(0, 2, 3, 1)
        global_pixel_coords[-1] = make_patch_coordinates(pos_h, pos_w, start=0, end=1)
        views[-1] = data["full_image"][0].to(device)
        vit_features[-1] = original_vit_features[0]
        extract_end_time = time.time()
        print(f"Feature extraction time: {extract_end_time - extract_start_time:.2f}s")
        denoising_start_time = time.time()
        denoised_results = denoise_an_image(
            args,
            all_raw_features=vit_features,
            all_pixel_coords=global_pixel_coords,
            all_transformed_views=views,
            device=device,
            img_pth=filename,
            denormalizer=denormalizer,
            should_save_vis=idx % args.vis_freq == 0,
        )
        num_processed_samples += 1
        denoising_end_time = time.time()
        print(f"Denoising time: {denoising_end_time - denoising_start_time:.2f}s")
        elapsed_time = time.time() - start_time
        eta = elapsed_time / (num_processed_samples) * (len(filenames) - num_processed_samples)
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        feature_extraction_time = extract_end_time - extract_start_time
        denoising_time = denoising_end_time - denoising_start_time
        feature_extraction_time_str = str(datetime.timedelta(seconds=int(feature_extraction_time)))
        denoising_time_str = str(datetime.timedelta(seconds=int(denoising_time)))
        print(
            f"[{idx+1}/{len(filenames)}] ETA: {eta_str}, Elapsed: {elapsed_str}, "
            f"Current Feature extraction time: {feature_extraction_time_str}, "
            f"Current Denoising time: {denoising_time_str}"
        )
        print("-" * 80)
    print(f"Total time: {elapsed_str}")


if __name__ == "__main__":
    args = get_args()
    main(args)
