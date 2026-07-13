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
from scipy.ndimage import zoom
from tqdm import tqdm

MODALITIES = ["t1", "t1ce", "t2", "flair"]
RAW_LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}

# Original BraTS volumes are 240x240x155 voxels. At full resolution, float16 storage
# for all 1251 cases needs ~100GB, which doesn't fit on disk-constrained environments
# like Colab's free tier (~110GB total, shared with the OS and raw downloads).
# Resampling to a fixed, smaller shape cuts this to a few GB while still preserving
# enough spatial detail for a portfolio-scale segmentation model. This also matches
# our training patch_size, so the whole volume becomes usable directly.
TARGET_SHAPE = (128, 128, 128)


def resample_volume(volume: np.ndarray, target_shape: tuple, order: int) -> np.ndarray:
    """Resample a (D, H, W) volume to target_shape.
    order=1 (linear) for images, order=0 (nearest) for label maps so we
    never invent new label values through interpolation."""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize using only nonzero (brain) voxels, since background is a large
    constant-zero region that would otherwise skew the mean/std."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume.astype(np.float32)
    mean, std = volume[mask].mean(), volume[mask].std()
    normalized = (volume - mean) / (std + 1e-8)
    normalized[~mask] = 0.0
    # float16 halves disk usage vs float32; precision loss is negligible for
    # normalized MRI intensities and standard for storage-constrained pipelines
    return normalized.astype(np.float16)


def remap_labels(label: np.ndarray) -> np.ndarray:
    # int8 is plenty for 4 label classes (0-3) and uses 1/8th the space of int64
    remapped = np.zeros_like(label, dtype=np.int8)
    for raw_val, new_val in RAW_LABEL_REMAP.items():
        remapped[label == raw_val] = new_val
    return remapped


def process_case(case_dir: Path, out_dir: Path, delete_raw_after: bool = False,
                  target_shape: tuple = TARGET_SHAPE) -> None:
    case_id = case_dir.name
    modality_volumes = []
    for mod in MODALITIES:
        path = case_dir / f"{case_id}_{mod}.nii.gz"
        volume = nib.load(str(path)).get_fdata()
        volume = resample_volume(volume, target_shape, order=1)
        modality_volumes.append(zscore_normalize(volume))
    image = np.stack(modality_volumes, axis=0)  # (4, *target_shape), float16

    seg_path = case_dir / f"{case_id}_seg.nii.gz"
    label = nib.load(str(seg_path)).get_fdata().astype(np.int64)
    label = resample_volume(label.astype(np.float32), target_shape, order=0).astype(np.int64)
    label = remap_labels(label)

    case_out_dir = out_dir / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_out_dir / "image.npy", image)
    np.save(case_out_dir / "label.npy", label)

    if delete_raw_after:
        # frees disk immediately so raw + processed never coexist at full scale —
        # essential on disk-constrained environments like Colab where both together
        # would exceed available space
        import shutil
        shutil.rmtree(case_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--delete_raw_after", action="store_true",
                         help="Delete each case's raw files immediately after processing it, "
                              "to avoid running out of disk on space-constrained environments.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    print(f"Found {len(case_dirs)} cases in {raw_dir}")

    for case_dir in tqdm(case_dirs, desc="Preprocessing"):
        # resume-safe: skip cases already processed (e.g. after a crash/restart)
        if (out_dir / case_dir.name / "label.npy").exists():
            continue
        try:
            process_case(case_dir, out_dir, delete_raw_after=args.delete_raw_after)
        except FileNotFoundError as e:
            print(f"Skipping {case_dir.name}: missing file ({e})")


if __name__ == "__main__":
    main()