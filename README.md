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

### Data

This project uses the [BraTS 2021 Task 1 dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1), available on Kaggle. It contains multi-institutional mpMRI scans (T1, T1ce, T2, FLAIR) with expert annotations for enhancing tumor, peritumoral edema, and necrotic tumor core.

1. Download the dataset from Kaggle (sign-in required)
2. Extract it into `data/raw/` so the structure looks like:
   ```
   data/raw/BraTS2021_Training_Data/
       BraTS2021_00000/
           BraTS2021_00000_t1.nii.gz
           BraTS2021_00000_t1ce.nii.gz
           BraTS2021_00000_t2.nii.gz
           BraTS2021_00000_flair.nii.gz
           BraTS2021_00000_seg.nii.gz
       BraTS2021_00002/
       ...
   ```
3. Preprocess into normalized numpy arrays:
   ```bash
   python -m src.data.preprocessing --raw_dir data/raw/BraTS2021_Training_Data --out_dir data/processed
   ```

Raw and processed data are excluded from version control via `.gitignore` — only code and results are tracked in this repo.

## Training

```bash
python -m src.train --config configs/unet_baseline.yaml
python -m src.train --config configs/attention_unet.yaml
```

## Evaluation

```bash
python -m src.evaluate --config configs/attention_unet.yaml --checkpoint checkpoints/attention_unet_best.pt
```

Reports Dice score, Hausdorff95, sensitivity, and specificity per tumor sub-region, plus qualitative overlays on held-out cases.

## Interactive demo

```bash
streamlit run demo/app.py
```

Select a case and slice, then view the model's predicted segmentation overlay against ground truth in real time.

## Detailed results

Full per-case metrics, training curves, and a discussion of failure cases (where the model under- or over-segments) are in [`docs/results.md`](docs/results.md). Including failure analysis on purpose — understanding where a model breaks matters as much as the headline score.

## What I'd improve with more time

- Test-time augmentation for more robust predictions
- 3D full-volume inference instead of patch-based (currently limited by GPU memory)
- Uncertainty estimation (e.g. MC dropout) to flag low-confidence predictions for clinician review

## References

- Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)," IEEE TMI 2015
- Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification," 2021
- Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," 2018
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015

## License

MIT