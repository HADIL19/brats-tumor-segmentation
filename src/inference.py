"""
Run inference on a single case and save an overlay visualization.

Usage:
    python src/inference.py --config configs/attention_unet.yaml \
        --checkpoint checkpoints/attention_unet_best.pt \
        --case_dir data/processed/BraTS20_Training_001 \
        --slice_idx 75
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.evaluate import sliding_window_inference
from src.train import build_model

CLASS_COLORS = {1: "yellow", 2: "green", 3: "red"}  # NCR/NET, ED, ET
CLASS_NAMES = {1: "NCR/NET", 2: "ED", 3: "ET"}


def overlay_mask(ax, base_slice: np.ndarray, mask_slice: np.ndarray):
    ax.imshow(base_slice, cmap="gray")
    for class_id, color in CLASS_COLORS.items():
        overlay = np.ma.masked_where(mask_slice != class_id, mask_slice)
        ax.imshow(overlay, cmap=plt.cm.colors.ListedColormap([color]), alpha=0.5, vmin=class_id, vmax=class_id)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--case_dir", type=str, required=True)
    parser.add_argument("--slice_idx", type=int, default=None, help="Defaults to the slice with the most tumor.")
    parser.add_argument("--out", type=str, default="assets/sample_predictions.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = torch.from_numpy(np.load(f"{args.case_dir}/image.npy")).float()
    label = np.load(f"{args.case_dir}/label.npy")

    patch_size = tuple(cfg["data"]["patch_size"])
    pred = sliding_window_inference(model, image, patch_size, device, cfg["model"]["num_classes"])

    slice_idx = args.slice_idx
    if slice_idx is None:
        tumor_per_slice = (label > 0).sum(axis=(1, 2))
        slice_idx = int(np.argmax(tumor_per_slice))

    base_slice = image[1, slice_idx].numpy()  # T1ce is a good anatomical background

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    overlay_mask(axes[0], base_slice, label[slice_idx])
    axes[0].set_title("Ground truth")
    overlay_mask(axes[1], base_slice, pred[slice_idx])
    axes[1].set_title("Prediction")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=CLASS_NAMES[i])
               for i, c in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {args.out}")


if __name__ == "__main__":
    main()
