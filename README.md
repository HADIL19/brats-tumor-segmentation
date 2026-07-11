# Brain Tumor Segmentation with Attention U-Net

Deep learning pipeline for multi-class brain tumor segmentation on MRI scans, built on the BraTS (Brain Tumor Segmentation) dataset. Implements a U-Net baseline and an Attention U-Net variant, with clinically relevant evaluation metrics and an interactive inference demo.

![demo](assets/demo.gif)

## Why this project

Tumor segmentation directly supports treatment planning — surgeons and oncologists use segmentation maps to estimate tumor volume, plan radiation fields, and track progression over time. Unlike simple classification, segmentation requires pixel-level precision and produces output that's directly usable in a clinical workflow.

This project segments three tumor sub-regions from multi-modal MRI:
- **Enhancing tumor (ET)**
- **Peritumoral edema (ED)**
- **Necrotic/non-enhancing tumor core (NCR/NET)**

## Results

| Model              | Dice (ET) | Dice (ED) | Dice (NCR/NET) | Mean Dice | Hausdorff95 (mm) |
|--------------------|-----------|-----------|------------------|-----------|-------------------|
| U-Net (baseline)   | 0.XX      | 0.XX      | 0.XX             | 0.XX      | X.XX              |
| Attention U-Net    | 0.XX      | 0.XX      | 0.XX             | 0.XX      | X.XX              |

*Fill in after training — see [Results](#detailed-results) for per-case breakdowns and failure analysis.*

![sample predictions](assets/sample_predictions.png)

## Architecture

- **Baseline:** 3D U-Net with instance normalization and residual blocks
- **Improved model:** Attention U-Net — attention gates at each skip connection let the decoder focus on relevant tumor regions rather than passing all encoder features through uniformly
- Trained on multi-modal input (T1, T1ce, T2, FLAIR stacked as channels)

See [`docs/architecture.md`](docs/architecture.md) for diagrams and design rationale.

## Repo structure

```
├── configs/              # YAML configs for each experiment
├── src/
│   ├── data/             # preprocessing, patch sampling, augmentation
│   ├── models/           # U-Net, Attention U-Net implementations
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/            # EDA and result exploration
├── demo/                 # Streamlit/Gradio inference app
├── docs/
│   └── architecture.md
├── assets/               # images/gifs for README
├── requirements.txt
└── Dockerfile
```

## Setup

```bash
git clone https://github.com/<your-username>/brats-tumor-segmentation.git
cd brats-tumor-segmentation
pip install -r requirements.txt
```

Data: download BraTS from the [official challenge site](https://www.med.upenn.edu/cbica/brats/) and place under `data/raw/`. See [`docs/data_prep.md`](docs/data_prep.md) for preprocessing steps.

## Training

```bash
python src/train.py --config configs/attention_unet.yaml
```

## Evaluation

```bash
python src/evaluate.py --checkpoint checkpoints/attention_unet_best.pt --split test
```

Reports Dice score, Hausdorff95, sensitivity, and specificity per tumor sub-region, plus qualitative overlays on held-out cases.

## Interactive demo

```bash
cd demo
streamlit run app.py
```

Upload an MRI slice (or use a provided sample) and view the predicted segmentation overlay in real time.

## Detailed results

Full per-case metrics, training curves, and a discussion of failure cases (where the model under- or over-segments) are in [`docs/results.md`](docs/results.md). Including failure analysis on purpose — understanding where a model breaks matters as much as the headline score.

## What I'd improve with more time

- Test-time augmentation for more robust predictions
- 3D full-volume inference instead of patch-based (currently limited by GPU memory)
- Uncertainty estimation (e.g. MC dropout) to flag low-confidence predictions for clinician review

## References

- Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)," IEEE TMI 2015
- Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," 2018
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015

## License

MIT
