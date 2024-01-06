from typing import Callable, Dict, Optional

import numpy as np
from torch import Tensor
from torchvision.datasets.folder import default_loader


class PairedListDataset:
    def __init__(
        self,
        data_list: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform
        self.data_list = data_list
        self.parse_data_list()
        self.loader = default_loader

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        img_pth = self.img_paths[index]
        img = self.transform(self.loader(img_pth))
        denoised_feats = np.load(self.denoised_feats[index]).squeeze()
        raw_vit_feats = np.load(self.raw_vit_feats[index]).squeeze()
        data_dict = {
            "image": img,
            "raw_vit_feats": raw_vit_feats,
            "denoised_feats": denoised_feats,
        }
        return data_dict

    def parse_data_list(self):
        # find valid samples from data_list
        self.img_paths = []
        self.raw_vit_feats = []
        self.denoised_feats = []
        with open(self.data_list, "r") as f:
            for line in f.readlines():
                lines = line.strip().split(" ")
                self.img_paths.append(lines[0])
                self.raw_vit_feats.append(lines[1])
                self.denoised_feats.append(lines[2])

    def __len__(self) -> int:
        return len(self.img_paths)
