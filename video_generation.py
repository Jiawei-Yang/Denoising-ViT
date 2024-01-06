# Modified from: https://github.com/facebookresearch/dino/blob/main/video_generation.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import glob
import sys
import argparse
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import DenoisingViT
from utils.visualization_tools import (
    get_pca_map,
    get_robust_pca,
)
import imageio
from torch_kmeans import KMeans, CosineSimilarity

FOURCC = {
    "mp4": cv2.VideoWriter_fourcc(*"MP4V"),
    "avi": cv2.VideoWriter_fourcc(*"XVID"),
}
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VideoGenerator:
    def __init__(self, args):
        self.args = args
        self.model = self.load_model()

    def run(self):
        if os.path.exists(self.args.input_path):
            frames_folder = os.path.join(
                os.path.dirname(self.args.output_path), "frames"
            )
            cluster_folder = os.path.join(self.args.output_path, "cluster_map")
            pca_folder = os.path.join(self.args.output_path, "pca_map")
            self.cluster_folder = cluster_folder
            self.pca_folder = pca_folder

            os.makedirs(cluster_folder, exist_ok=True)
            os.makedirs(pca_folder, exist_ok=True)
            os.makedirs(frames_folder, exist_ok=True)

            self.extract_frames_from_video(self.args.input_path, frames_folder)

            self.inference(frames_folder)

            self.generate_video_from_images()
        else:
            print(f"Provided input path {self.args.input_path} doesn't exists.")
            sys.exit(1)

    def extract_frames_from_video(self, inp: str, out: str):
        vidcap = cv2.VideoCapture(inp)
        self.args.fps = vidcap.get(cv2.CAP_PROP_FPS)

        print(f"Video: {inp} ({self.args.fps} fps)")
        print(f"Extracting frames to {out}")

        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(
                os.path.join(out, f"frame-{count:04}.jpg"),
                image,
            )
            success, image = vidcap.read()
            count += 1

    def generate_video_from_images(self):
        folder_name = ["cluster_map", "pca_map"]
        for i, folder in enumerate([self.cluster_folder, self.pca_folder]):
            for feat_type in ["original", "denoised"]:
                img_array = []
                img_list = sorted(glob.glob(os.path.join(folder, f"{feat_type}-*.jpg")))
                print(
                    f"saving video to {self.args.output_path}/video_{feat_type}_{folder_name[i]}.{self.args.video_format}"
                )
                for filename in tqdm(img_list):
                    with open(filename, "rb") as f:
                        img = Image.open(f)
                        img = img.convert("RGB")
                        img_array.append(img)

                imageio.mimsave(
                    os.path.join(
                        self.args.output_path,
                        f"video_{feat_type}_{folder_name[i]}." + self.args.video_format,
                    ),
                    img_array,
                    fps=self.args.fps,
                )
        print("Done")

    def inference(self, inp: str):
        cache_to_compute_pca_raw = []
        cache_to_compute_pca_denoised = []
        n_clusters = self.args.n_clusters
        kmeans_raw = KMeans(n_clusters=n_clusters, distance=CosineSimilarity)
        kmeans_denoised = KMeans(n_clusters=n_clusters, distance=CosineSimilarity)
        all_files = sorted(glob.glob(os.path.join(inp, "*.jpg")))
        interested_files = all_files[:30]  # + all_files[-30:]
        # cmap = plt.get_cmap("rainbow", n_clusters)
        # cmap = plt.get_cmap("tab20", n_clusters)
        cmap = plt.get_cmap("tab20")
        for i, img_path in tqdm(
            enumerate(interested_files),
            desc="Computing PCA",
            total=len(interested_files),
        ):
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")

            img = self.transform(img)

            # make the image divisible by the patch size
            w, h = (
                img.shape[1] - img.shape[1] % self.args.vit_stride,
                img.shape[2] - img.shape[2] % self.args.vit_stride,
            )
            img = img[:, :w, :h].unsqueeze(0)

            output_dict = self.model(img.to(DEVICE), return_dict=True)
            raw_vit_feats = output_dict["raw_vit_feats"]
            raw_vit_feats = raw_vit_feats.reshape(1, -1, raw_vit_feats.shape[-1])
            random_patches = torch.randperm(raw_vit_feats.shape[1])  # [:100]
            raw_vit_feats = raw_vit_feats[:, random_patches, :]
            kmeans_raw.fit(raw_vit_feats)
            cache_to_compute_pca_raw.append(raw_vit_feats)
            denoised_features = output_dict["pred_denoised_feats"]
            kmeans_denoised.fit(
                denoised_features.reshape(1, -1, denoised_features.shape[-1])
            )
            cache_to_compute_pca_denoised.append(denoised_features)
        all_raw = torch.cat(cache_to_compute_pca_raw, dim=0)
        reduct_mat, color_min, color_max = get_robust_pca(
            all_raw.reshape(-1, all_raw.shape[-1]), m=2.5
        )
        raw_pca_stats = (reduct_mat, color_min, color_max)
        all_denoised = torch.cat(cache_to_compute_pca_denoised, dim=0)
        _reduct_mat, _color_min, _color_max = get_robust_pca(
            all_denoised.reshape(-1, all_denoised.shape[-1]), m=2.5
        )
        denoised_pca_stats = (_reduct_mat, _color_min, _color_max)
        for img_path in tqdm(
            sorted(glob.glob(os.path.join(inp, "*.jpg")))[:: self.args.subsample_ratio]
        ):
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")

            img = self.transform(img)

            # make the image divisible by the patch size
            w, h = (
                img.shape[1] - img.shape[1] % self.args.vit_stride,
                img.shape[2] - img.shape[2] % self.args.vit_stride,
            )
            img = img[:, :w, :h].unsqueeze(0)
            output_dict = self.model(img.to(DEVICE), return_dict=True)
            raw_vit_feats = output_dict["raw_vit_feats"]
            denoised_features = output_dict["pred_denoised_feats"]
            raw_feat_color = get_pca_map(
                raw_vit_feats, [img.shape[-2], img.shape[-1]], pca_stats=raw_pca_stats
            )
            fname = os.path.join(
                self.pca_folder, "original-" + os.path.basename(img_path)
            )
            raw_feat_color = np.uint8(raw_feat_color * 255)
            image = Image.fromarray(raw_feat_color)
            image.save(fname)

            labels = kmeans_raw.predict(
                raw_vit_feats.reshape(1, -1, raw_vit_feats.shape[-1])
            )
            labels = labels.reshape(
                1, raw_vit_feats.shape[-3], raw_vit_feats.shape[-2]
            ).float()
            labels = (
                nn.functional.interpolate(
                    labels.unsqueeze(0),
                    size=(img.shape[-2], img.shape[-1]),
                    mode="nearest",
                )[0][0]
                .cpu()
                .numpy()
            )
            label_map = cmap(labels / n_clusters)[..., :3]
            fname = os.path.join(
                self.cluster_folder, "original-" + os.path.basename(img_path)
            )
            label_map = np.uint8(label_map * 255)
            image = Image.fromarray(label_map)
            image.save(fname)

            denoised_feat_color = get_pca_map(
                denoised_features,
                [img.shape[-2], img.shape[-1]],
                pca_stats=denoised_pca_stats,
            )
            denoised_feat_color = np.uint8(denoised_feat_color * 255)
            fname = os.path.join(
                self.pca_folder, "denoised-" + os.path.basename(img_path)
            )
            image = Image.fromarray(denoised_feat_color)
            image.save(fname)

            labels = kmeans_denoised.predict(
                denoised_features.reshape(1, -1, denoised_features.shape[-1])
            )
            labels = labels.reshape(
                1, denoised_features.shape[-3], denoised_features.shape[-2]
            ).float()
            labels = (
                nn.functional.interpolate(
                    labels.unsqueeze(0),
                    size=(img.shape[-2], img.shape[-1]),
                    mode="nearest",
                )[0][0]
                .cpu()
                .numpy()
            )
            label_map = cmap(labels / n_clusters)[..., :3]
            fname = os.path.join(
                self.cluster_folder, "denoised-" + os.path.basename(img_path)
            )
            label_map = np.uint8(label_map * 255)
            image = Image.fromarray(label_map)
            image.save(fname)

    def load_model(self):
        # build model
        vit = DenoisingViT.ViTWrapper(
            model_type=self.args.vit_type,
            stride=self.args.vit_stride,
        )
        vit = vit.to(DEVICE)
        model = DenoisingViT.Denoiser(
            noise_map_height=self.args.noise_map_height,
            noise_map_width=self.args.noise_map_width,
            feature_dim=vit.n_output_dims,
            vit=vit,
            enable_pe=self.args.enable_pe,
        ).to(DEVICE)
        if args.load_denoiser_from is not None:
            freevit_model_ckpt = torch.load(args.load_denoiser_from)["denoiser"]
            msg = model.load_state_dict(freevit_model_ckpt, strict=False)
        for k in model.state_dict().keys():
            if k in freevit_model_ckpt:
                print(k, "loaded")
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(DEVICE)
        normalizer = vit.transformation.transforms[-1]
        if self.args.resize is not None:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(self.args.resize), normalizer]
            )
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalizer])
        return model


def parse_args():
    parser = argparse.ArgumentParser("Generation self-attention video")
    parser.add_argument(
        "--enable_pe",
        action="store_true",
    )
    parser.add_argument(
        "--vit_type",
        default="vit_base_patch14_dinov2.lvd142m",
        type=str,
    )
    parser.add_argument(
        "--vit_stride", default=14, type=int, help="Patch resolution of the self.model."
    )
    parser.add_argument(
        "--n_clusters", default=20, type=int, help="Patch resolution of the self.model."
    )
    parser.add_argument(
        "--subsample_ratio", default=1, type=int, help="stride to sample from the video"
    )
    parser.add_argument("--noise_map_height", default=37, type=int)
    parser.add_argument("--noise_map_width", default=37, type=int)
    parser.add_argument("--load_denoiser_from", default=None, type=str)
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", default="./", type=str)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx percent of the mass.""",
    )
    parser.add_argument(
        "--resize",
        default=None,
        type=int,
        nargs="+",
        help="""Apply a resize transformation to input image(s). Use if OOM error.
        Usage (single or H W): --resize 512, --resize 720 1280""",
    )
    parser.add_argument(
        "--fps",
        default=30.0,
        type=float,
        help="FPS of input / output video. Automatically set if you extract frames from a video.",
    )
    parser.add_argument(
        "--video_format",
        default="mp4",
        type=str,
        choices=["mp4", "avi"],
        help="Format of generated video (mp4 or avi).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vg = VideoGenerator(args)
    vg.run()
