"""
Interactive Streamlit demo for tumor segmentation.

Run with: streamlit run demo/app.py

Lets a user pick a preprocessed case, view a slice, and see the model's
segmentation overlay compared to ground truth. This is the single highest
impact piece of the whole repo for a recruiter skimming quickly — it turns
"a notebook with metrics" into "a thing I can click and see work."
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluate import sliding_window_inference
from src.train import build_model

CLASS_COLORS = {1: "yellow", 2: "green", 3: "red"}
CLASS_NAMES = {1: "NCR/NET (necrotic core)", 2: "ED (edema)", 3: "ET (enhancing tumor)"}

st.set_page_config(page_title="Brain Tumor Segmentation Demo", layout="wide")
st.title("Brain Tumor Segmentation — Attention U-Net Demo")
st.caption("Trained on BraTS. Select a case and slice to view the model's predicted segmentation.")


@st.cache_resource
def load_model(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg, device


def overlay_mask(ax, base_slice, mask_slice, title):
    ax.imshow(base_slice, cmap="gray")
    for class_id, color in CLASS_COLORS.items():
        overlay = np.ma.masked_where(mask_slice != class_id, mask_slice)
        ax.imshow(overlay, cmap=plt.cm.colors.ListedColormap([color]), alpha=0.5, vmin=class_id, vmax=class_id)
    ax.set_title(title)
    ax.axis("off")


config_path = st.sidebar.text_input("Config path", "configs/attention_unet.yaml")
checkpoint_path = st.sidebar.text_input("Checkpoint path", "checkpoints/attention_unet_best.pt")
data_dir = st.sidebar.text_input("Processed data dir", "data/processed")

if not Path(checkpoint_path).exists():
    st.warning(f"Checkpoint not found at `{checkpoint_path}`. Train a model first with `python src/train.py`.")
    st.stop()

model, cfg, device = load_model(config_path, checkpoint_path)

case_ids = sorted([p.name for p in Path(data_dir).iterdir() if p.is_dir()]) if Path(data_dir).exists() else []
if not case_ids:
    st.warning(f"No preprocessed cases found in `{data_dir}`. Run `src/data/preprocessing.py` first.")
    st.stop()

case_id = st.sidebar.selectbox("Case", case_ids)
image = torch.from_numpy(np.load(f"{data_dir}/{case_id}/image.npy")).float()
label = np.load(f"{data_dir}/{case_id}/label.npy")

slice_idx = st.sidebar.slider("Slice index", 0, image.shape[1] - 1, image.shape[1] // 2)

if st.sidebar.button("Run segmentation", type="primary"):
    with st.spinner("Running inference..."):
        patch_size = tuple(cfg["data"]["patch_size"])
        pred = sliding_window_inference(model, image, patch_size, device, cfg["model"]["num_classes"])

    base_slice = image[1, slice_idx].numpy()

    col1, col2 = st.columns(2)
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    overlay_mask(ax1, base_slice, label[slice_idx], "Ground truth")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    overlay_mask(ax2, base_slice, pred[slice_idx], "Prediction")
    col2.pyplot(fig2)

    st.markdown("**Legend:** " + " | ".join(f":{v.split()[0].lower()}[{v}]" for v in CLASS_NAMES.values()))
else:
    st.info("Adjust the slice index and click **Run segmentation** to see the model's prediction.")
