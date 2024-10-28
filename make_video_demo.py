import argparse
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch_kmeans import CosineSimilarity, KMeans
from tqdm import tqdm

import dvt.models as DVT
from dvt.utils.visualization import get_robust_pca

vit_type = "vit_base_patch14_dinov2.lvd142m"
stride = 4
h, w = 490, 854
patch_size = 14
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = (
    DVT.PretrainedViTWrapper(model_identifier=vit_type, stride=stride, patch_size=patch_size)
    .to(device)
    .eval()
)
ckpt_pth = "ckpts/imgnet_distilled/vit_base_patch14_dinov2.lvd142m.pth"
ckpt = torch.load(ckpt_pth, map_location=device)["model"]
new_state_dict = ckpt
msg = vit.load_state_dict(new_state_dict, strict=True)
print(msg)

normalizer = vit.transformation.transforms[-1]
denormalizer = transforms.Normalize(
    mean=[-m / s for m, s in zip(normalizer.mean, normalizer.std)],
    std=[1 / s for s in normalizer.std],
)

base_transform = transforms.Compose(
    [
        transforms.Resize((h, w), Image.BICUBIC, antialias=True),
        transforms.ToTensor(),
        normalizer,
    ]
)

output_path = "work_dirs/davis_demo/"
image_paths = ["demo/davis-mallard-water"]

s_id = 0
for scene in tqdm(image_paths):
    image_path = image_paths[s_id]
    frame_paths = os.listdir(image_path)
    frame_paths.sort()
    images = []
    for frame in frame_paths:
        image = Image.open(os.path.join(image_path, frame))
        images.append(image.convert("RGB"))
    s_id += 1
    (
        image_video,
        instance_pca_video,
        dataset_pca_video,
        kmeans_video,
        first_pca_video,
        second_pca_video,
        third_pca_video,
        fg_pca_video,
        norm_video,
        fg_pca_standard_video,
    ) = ([], [], [], [], [], [], [], [], [], [])
    pbar = tqdm(enumerate(images), total=len(images))
    for i, img in pbar:
        img = base_transform(img).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            feat = vit.get_intermediate_layers(
                img,
                n=[vit.last_layer_index],
                reshape=True,
                norm=True,
            )[-1].permute(0, 2, 3, 1)

        pbar.set_description(f"Processing {scene} [{s_id}/{len(image_paths)}, {i}/{len(images)}]")

        # 1. input image
        img = denormalizer(img.squeeze(0))
        img = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        os.makedirs(os.path.join(output_path, scene, "images"), exist_ok=True)
        img = Image.fromarray(img)
        img.save(os.path.join(output_path, f"{scene}/images/{i:02d}_input.png"))
        image_video.append(img)

        if i == 0:
            # 2. instance pca
            instance_reduct_mat, _, _ = get_robust_pca(feat.reshape(-1, feat.shape[-1]), m=2)
            # 3. dataset pca
            stats = torch.load("demo/assets/stats.pth")
            mat = stats["denoised_reduct_mat_full"]
            dataset_reduct_mat = mat.to(device)

            mat_standard = stats["denoised_standard_mapping"]
            dataset_reduct_mat_standard = mat_standard.to(device)

            kmeans = KMeans(n_clusters=8, device=device, distance=CosineSimilarity)
            kmeans.fit(feat.reshape(1, -1, feat.shape[-1]))
            cmap = plt.get_cmap("rainbow")

            inferno_cmap = plt.get_cmap("inferno")

        # 2. instance pca
        pca = torch.matmul(feat.reshape(-1, feat.shape[-1]), instance_reduct_mat)
        pca = (pca - pca.min(dim=0, keepdim=True)[0]) / (
            pca.max(dim=0, keepdim=True)[0] - pca.min(dim=0, keepdim=True)[0]
        )
        pca = pca.reshape(120, 211, 3).cpu().numpy()
        pca = (pca * 255).astype(np.uint8)
        pca = Image.fromarray(pca).resize((w, h), Image.BICUBIC)
        pca.save(os.path.join(output_path, f"{scene}/images/{i:02d}_pca_instance.png"))
        instance_pca_video.append(pca)

        # 3. dataset pca
        pca_full = torch.matmul(feat.reshape(-1, feat.shape[-1]), dataset_reduct_mat)
        pca = (pca_full - pca_full.min(dim=0, keepdim=True)[0]) / (
            pca_full.max(dim=0, keepdim=True)[0] - pca_full.min(dim=0, keepdim=True)[0]
        )
        pca = pca.reshape(120, 211, 3).cpu().numpy()
        pca = (pca * 255).astype(np.uint8)
        pca = Image.fromarray(pca).resize((w, h), Image.BICUBIC)
        pca.save(os.path.join(output_path, f"{scene}/images/{i:02d}_pca_dataset.png"))
        dataset_pca_video.append(pca)

        # 4. kmeans
        kmeans_labels = kmeans.predict(feat.reshape(1, -1, feat.shape[-1])).float()
        kmeans_labels = kmeans_labels.reshape(120, 211, 1).cpu().numpy()
        kmeans_cluster_map = cmap(kmeans_labels.squeeze(2) / 8)[:, :, :3]
        kmeans_cluster_map = (kmeans_cluster_map * 255).astype(np.uint8)
        kmeans_cluster_map = Image.fromarray(kmeans_cluster_map).resize((w, h), Image.BICUBIC)
        kmeans_cluster_map.save(os.path.join(output_path, f"{scene}/images/{i:02d}_kmeans.png"))
        kmeans_video.append(kmeans_cluster_map)

        # 5. 1st pca
        first_pca = pca_full[..., 0]
        first_pca = (first_pca - first_pca.min()) / (first_pca.max() - first_pca.min())
        first_pca = first_pca.reshape(120, 211).cpu().numpy()
        first_pca = inferno_cmap(first_pca)[:, :, :3]
        first_pca = (first_pca * 255).astype(np.uint8)
        first_pca = Image.fromarray(first_pca).resize((w, h), Image.BICUBIC)
        first_pca.save(os.path.join(output_path, f"{scene}/images/{i:02d}_first_pca.png"))
        first_pca_video.append(first_pca)

        # 6. 2st pca
        second_pca = 1 - pca_full[..., 1]
        second_pca = (second_pca - second_pca.min()) / (second_pca.max() - second_pca.min())
        second_pca = second_pca.reshape(120, 211).cpu().numpy()
        second_pca = inferno_cmap(second_pca)[:, :, :3]
        second_pca = (second_pca * 255).astype(np.uint8)
        second_pca = Image.fromarray(second_pca).resize((w, h), Image.BICUBIC)
        second_pca.save(os.path.join(output_path, f"{scene}/images/{i:02d}_second_pca.png"))
        second_pca_video.append(second_pca)

        # 7. 3st pca
        third_pca = pca_full[..., 2]
        third_pca = (third_pca - third_pca.min()) / (third_pca.max() - third_pca.min())
        third_pca = third_pca.reshape(120, 211).cpu().numpy()
        third_pca = inferno_cmap(third_pca)[:, :, :3]
        third_pca = (third_pca * 255).astype(np.uint8)
        third_pca = Image.fromarray(third_pca).resize((w, h), Image.BICUBIC)
        third_pca.save(os.path.join(output_path, f"{scene}/images/{i:02d}_third_pca.png"))
        third_pca_video.append(third_pca)

        # 8. fg pca
        fg_mask = 1 - pca_full[..., 1] > 0.1
        if i == 0:
            fg_feats = feat.reshape(-1, feat.shape[-1])[fg_mask]
            fg_pca_reduct = torch.pca_lowrank(fg_feats, q=3, niter=20)[2]
        fg_pca = torch.matmul(feat.reshape(-1, feat.shape[-1]), fg_pca_reduct)
        fg_pca = (fg_pca - fg_pca.min(dim=0, keepdim=True)[0]) / (
            fg_pca.max(dim=0, keepdim=True)[0] - fg_pca.min(dim=0, keepdim=True)[0]
        )
        fg_pca = fg_pca.reshape(120, 211, 3).cpu().numpy()
        fg_pca = fg_pca * fg_mask.reshape(120, 211, 1).cpu().numpy()
        fg_pca = (fg_pca * 255).astype(np.uint8)
        fg_pca = Image.fromarray(fg_pca).resize((w, h), Image.BICUBIC)
        fg_pca.save(os.path.join(output_path, f"{scene}/images/{i:02d}_fg_pca.png"))
        fg_pca_video.append(fg_pca)

        # 9. fg pca
        fg_mask = (feat.view(-1, 768) @ dataset_reduct_mat_standard > 0).view(-1)
        if i == 0:
            fg_feat2s = feat.reshape(-1, feat.shape[-1])[fg_mask]
            fg_pca_reduct2 = torch.pca_lowrank(fg_feat2s, q=3, niter=20)[2]
        fg_pca = torch.matmul(feat.reshape(-1, feat.shape[-1]), fg_pca_reduct2)
        fg_pca = (fg_pca - fg_pca.min(dim=0, keepdim=True)[0]) / (
            fg_pca.max(dim=0, keepdim=True)[0] - fg_pca.min(dim=0, keepdim=True)[0]
        )
        fg_pca = fg_pca.reshape(120, 211, 3).cpu().numpy()
        fg_pca = fg_pca * fg_mask.reshape(120, 211, 1).cpu().numpy()
        fg_pca = (fg_pca * 255).astype(np.uint8)
        fg_pca = Image.fromarray(fg_pca).resize((w, h), Image.BICUBIC)
        fg_pca.save(os.path.join(output_path, f"{scene}/images/{i:02d}_fg_pca_standard.png"))
        fg_pca_standard_video.append(fg_pca)

        # 9 feat norm
        norm = torch.norm(feat, dim=-1, keepdim=True)
        norm = F.softmax(norm.reshape(1, -1) / 5, dim=-1).reshape(norm.shape)
        norm = (norm - norm.min()) / (norm.max() - norm.min())
        norm = norm.reshape(120, 211).cpu().numpy()
        norm = inferno_cmap(norm)[:, :, :3]
        norm = (norm * 255).astype(np.uint8)
        norm = Image.fromarray(norm).resize((w, h), Image.BICUBIC)
        norm.save(os.path.join(output_path, f"{scene}/images/{i:02d}_norm.png"))
        norm_video.append(norm)

    imageio.mimsave(os.path.join(output_path, f"{scene}/image.mp4"), image_video, fps=20)
    imageio.mimsave(
        os.path.join(output_path, f"{scene}/instance_pca.mp4"),
        instance_pca_video,
        fps=20,
    )
    imageio.mimsave(
        os.path.join(output_path, f"{scene}/dataset_pca.mp4"), dataset_pca_video, fps=20
    )
    imageio.mimsave(os.path.join(output_path, f"{scene}/kmeans.mp4"), kmeans_video, fps=20)
    imageio.mimsave(os.path.join(output_path, f"{scene}/first_pca.mp4"), first_pca_video, fps=20)
    imageio.mimsave(os.path.join(output_path, f"{scene}/second_pca.mp4"), second_pca_video, fps=20)
    imageio.mimsave(os.path.join(output_path, f"{scene}/third_pca.mp4"), third_pca_video, fps=20)
    imageio.mimsave(os.path.join(output_path, f"{scene}/fg_pca.mp4"), fg_pca_video, fps=20)
    imageio.mimsave(os.path.join(output_path, f"{scene}/norm.mp4"), norm_video, fps=20)
    imageio.mimsave(
        os.path.join(output_path, f"{scene}/fg_pca_standard.mp4"),
        fg_pca_standard_video,
        fps=20,
    )
