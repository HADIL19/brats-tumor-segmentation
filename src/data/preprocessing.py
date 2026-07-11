"""
Preprocess raw BraTS NIfTI files into normalized numpy arrays for training.

Raw BraTS case directory is expected to look like:
  BraTS20_Training_XXX/
      BraTS20_Training_XXX_t1.nii.gz
      BraTS20_Training_XXX_t1ce.nii.gz
      BraTS20_Training_XXX_t2.nii.gz
      BraTS20_Training_XXX_flair.nii.gz
      BraTS20_Training_XXX_seg.nii.gz

Raw BraTS labels: 0=background, 1=NCR/NET, 2=ED, 4=ET.
We remap 4 -> 3 so labels are contiguous {0, 1, 2, 3}, which is required
for one-hot encoding and standard loss functions.

Usage:
    python src/data/preprocessing.py --raw_dir data/raw --out_dir data/processed
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

MODALITIES = ["t1", "t1ce", "t2", "flair"]
RAW_LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize using only nonzero (brain) voxels, since background is a large
    constant-zero region that would otherwise skew the mean/std."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume.astype(np.float32)
    mean, std = volume[mask].mean(), volume[mask].std()
    normalized = (volume - mean) / (std + 1e-8)
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)


def remap_labels(label: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(label, dtype=np.int64)
    for raw_val, new_val in RAW_LABEL_REMAP.items():
        remapped[label == raw_val] = new_val
    return remapped


def process_case(case_dir: Path, out_dir: Path) -> None:
    case_id = case_dir.name
    modality_volumes = []
    for mod in MODALITIES:
        path = case_dir / f"{case_id}_{mod}.nii.gz"
        volume = nib.load(str(path)).get_fdata()
        modality_volumes.append(zscore_normalize(volume))
    image = np.stack(modality_volumes, axis=0)  # (4, D, H, W)

    seg_path = case_dir / f"{case_id}_seg.nii.gz"
    label = nib.load(str(seg_path)).get_fdata().astype(np.int64)
    label = remap_labels(label)

    case_out_dir = out_dir / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_out_dir / "image.npy", image)
    np.save(case_out_dir / "label.npy", label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    print(f"Found {len(case_dirs)} cases in {raw_dir}")

    for case_dir in tqdm(case_dirs, desc="Preprocessing"):
        try:
            process_case(case_dir, out_dir)
        except FileNotFoundError as e:
            print(f"Skipping {case_dir.name}: missing file ({e})")


if __name__ == "__main__":
    main()
