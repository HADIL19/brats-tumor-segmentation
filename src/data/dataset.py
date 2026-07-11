"""
BraTS dataset loader.

Expects preprocessed data laid out as:
  data/processed/<case_id>/
      image.npy   -> shape (4, D, H, W), float32, z-score normalized, channels = [T1, T1ce, T2, FLAIR]
      label.npy   -> shape (D, H, W), int64, values in {0, 1, 2, 3}
                     (0=background, 1=NCR/NET, 2=ED, 3=ET — remapped from raw BraTS labels)

Run src/data/preprocessing.py first to convert raw NIfTI files into this format.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, data_dir: str, case_ids: list[str], patch_size: tuple[int, int, int] = (128, 128, 128),
                 train: bool = True):
        self.data_dir = Path(data_dir)
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.train = train

    def __len__(self) -> int:
        return len(self.case_ids)

    def _random_crop(self, image: np.ndarray, label: np.ndarray):
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size

        # bias crop location toward tumor-containing regions half the time,
        # since most of the volume is background and random crops would mostly miss the tumor
        if self.train and random.random() < 0.5 and label.max() > 0:
            tumor_voxels = np.argwhere(label > 0)
            center = tumor_voxels[random.randint(0, len(tumor_voxels) - 1)]
            start_d = np.clip(center[0] - pd // 2, 0, max(d - pd, 0))
            start_h = np.clip(center[1] - ph // 2, 0, max(h - ph, 0))
            start_w = np.clip(center[2] - pw // 2, 0, max(w - pw, 0))
        else:
            start_d = random.randint(0, max(d - pd, 0))
            start_h = random.randint(0, max(h - ph, 0))
            start_w = random.randint(0, max(w - pw, 0))

        image_crop = image[:, start_d:start_d + pd, start_h:start_h + ph, start_w:start_w + pw]
        label_crop = label[start_d:start_d + pd, start_h:start_h + ph, start_w:start_w + pw]
        return image_crop, label_crop

    def _augment(self, image: np.ndarray, label: np.ndarray):
        # random axis flips — cheap, effective, and anatomically sensible for MRI
        for axis in (1, 2, 3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis - 1).copy()
        return image, label

    def __getitem__(self, idx: int):
        case_id = self.case_ids[idx]
        case_dir = self.data_dir / case_id
        image = np.load(case_dir / "image.npy")
        label = np.load(case_dir / "label.npy")

        image, label = self._random_crop(image, label)
        if self.train:
            image, label = self._augment(image, label)

        return torch.from_numpy(image.copy()).float(), torch.from_numpy(label.copy()).long()


def get_case_ids(data_dir: str) -> list[str]:
    """List all preprocessed case IDs available under data_dir."""
    return sorted([p.name for p in Path(data_dir).iterdir() if p.is_dir()])
