"""
Evaluation script — runs the trained model over held-out cases and reports
per-class Dice, Hausdorff95, sensitivity, and specificity.

Usage:
    python src/evaluate.py --config configs/attention_unet.yaml --checkpoint checkpoints/attention_unet_best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataset import get_case_ids
from src.metrics import evaluate_case
from src.train import MODEL_REGISTRY, build_model

CLASS_NAMES = {1: "NCR/NET", 2: "ED", 3: "ET"}


def sliding_window_inference(model, image: torch.Tensor, patch_size: tuple, device, num_classes: int) -> np.ndarray:
    """
    Run inference over a full volume via overlapping patches, averaging logits
    in overlap regions. Needed because full BraTS volumes (240x240x155) are
    larger than what fits in GPU memory at training patch size.
    """
    _, d, h, w = image.shape
    pd, ph, pw = patch_size
    stride = (pd // 2, ph // 2, pw // 2)

    logits_sum = torch.zeros((num_classes, d, h, w))
    counts = torch.zeros((1, d, h, w))

    model.eval()
    with torch.no_grad():
        for z in range(0, max(d - pd, 0) + 1, stride[0]):
            for y in range(0, max(h - ph, 0) + 1, stride[1]):
                for x in range(0, max(w - pw, 0) + 1, stride[2]):
                    z_end, y_end, x_end = min(z + pd, d), min(y + ph, h), min(x + pw, w)
                    patch = image[:, z:z_end, y:y_end, x:x_end].unsqueeze(0).to(device)

                    pad = (0, pw - patch.shape[-1], 0, ph - patch.shape[-2], 0, pd - patch.shape[-3])
                    patch_padded = torch.nn.functional.pad(patch, pad)

                    out = model(patch_padded).cpu().squeeze(0)
                    out = out[:, :z_end - z, :y_end - y, :x_end - x]

                    logits_sum[:, z:z_end, y:y_end, x:x_end] += out
                    counts[:, z:z_end, y:y_end, x:x_end] += 1

    avg_logits = logits_sum / counts.clamp(min=1)
    return avg_logits.argmax(dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="docs/results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    case_ids = get_case_ids(cfg["data"]["processed_dir"])
    _, val_ids = __import__("src.train", fromlist=["split_cases"]).split_cases(
        case_ids, val_fraction=cfg["data"].get("val_fraction", 0.2))

    patch_size = tuple(cfg["data"]["patch_size"])
    all_results = {}

    for case_id in val_ids:
        case_dir = Path(cfg["data"]["processed_dir"]) / case_id
        image = torch.from_numpy(np.load(case_dir / "image.npy")).float()
        label = np.load(case_dir / "label.npy")

        pred = sliding_window_inference(model, image, patch_size, device, cfg["model"]["num_classes"])
        all_results[case_id] = evaluate_case(pred, label, CLASS_NAMES)
        print(f"{case_id}: " + ", ".join(f"{k} Dice={v['dice']:.3f}" for k, v in all_results[case_id].items()))

    # aggregate mean per class
    summary = {}
    for name in CLASS_NAMES.values():
        dices = [r[name]["dice"] for r in all_results.values()]
        summary[name] = {"mean_dice": float(np.mean(dices)), "std_dice": float(np.std(dices))}

    print("\n--- Summary ---")
    for name, stats in summary.items():
        print(f"{name}: Dice = {stats['mean_dice']:.4f} +/- {stats['std_dice']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_case": all_results, "summary": summary}, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
