import os
from typing import Callable, Dict, Optional

import numpy as np
from torch import Tensor
from torchvision.datasets.folder import default_loader


class PairedListDataset:
    def __init__(
        self,
        data_root: str,
        data_list: str,
        feat_root: str,
        transform: Optional[Callable] = None,
    ):
        self.data_root = data_root
        self.feat_root = feat_root
        self.transform = transform
        self.data_list = data_list

        with open(self.data_list, "r") as f:
            lines = f.readlines()
        self.img_paths = [line.strip().split(" ")[0] for line in lines]
        self.loader = default_loader

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        img_pth = self.img_paths[index]
        ext = os.path.splitext(img_pth)[1]
        denoised_feats_pth = os.path.join(self.feat_root, img_pth.replace(f"{ext}", ".npy"))
        if not os.path.exists(denoised_feats_pth):
            return self.__getitem__(np.random.randint(len(self.img_paths)))
        original_feats_pth = denoised_feats_pth.replace("denoised_features", "raw_features")
        img_pth = os.path.join(self.data_root, img_pth)
        img = self.transform(self.loader(img_pth))
        denoised_feats = np.load(denoised_feats_pth).squeeze()
        original_feats = np.load(original_feats_pth).squeeze()
        data_dict = {
            "image": img,
            "original_feats": original_feats,
            "denoised_feats": denoised_feats,
        }
        return data_dict

    def __len__(self) -> int:
        return len(self.img_paths)
