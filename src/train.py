"""
Training script for BraTS tumor segmentation.

Usage:
    python src/train.py --config configs/attention_unet.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BraTSDataset, get_case_ids
from src.models.attention_unet import AttentionUNet3D
from src.models.losses import DiceCELoss
from src.models.unet import UNet3D

MODEL_REGISTRY = {
    "unet": UNet3D,
    "attention_unet": AttentionUNet3D,
}


def build_model(cfg: dict) -> torch.nn.Module:
    model_cls = MODEL_REGISTRY[cfg["model"]["name"]]
    return model_cls(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"].get("base_channels", 32),
    )


def split_cases(case_ids: list[str], val_fraction: float = 0.2, seed: int = 42):
    import random
    rng = random.Random(seed)
    shuffled = case_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    case_ids = get_case_ids(cfg["data"]["processed_dir"])
    train_ids, val_ids = split_cases(case_ids, val_fraction=cfg["data"].get("val_fraction", 0.2))
    print(f"Train cases: {len(train_ids)} | Val cases: {len(val_ids)}")

    patch_size = tuple(cfg["data"]["patch_size"])
    train_ds = BraTSDataset(cfg["data"]["processed_dir"], train_ids, patch_size, train=True)
    val_ds = BraTSDataset(cfg["data"]["processed_dir"], val_ids, patch_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               num_workers=cfg["train"].get("num_workers", 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["train"].get("num_workers", 4), pin_memory=True)

    model = build_model(cfg).to(device)
    criterion = DiceCELoss(num_classes=cfg["model"]["num_classes"], dice_weight=cfg["train"].get("dice_weight", 0.5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                   weight_decay=cfg["train"].get("weight_decay", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{cfg['train']['epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / f"{cfg['model']['name']}_best.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_path)
            print(f"  Saved new best checkpoint to {ckpt_path} (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
