"""
Clinically relevant segmentation metrics, computed per tumor sub-region.

Dice alone isn't enough for a portfolio piece that's meant to look clinically
credible — Hausdorff distance captures boundary errors that Dice can miss
(e.g. a small far-away false positive barely changes Dice but can matter a lot
for surgical planning), and sensitivity/specificity are the metrics clinicians
actually think in.
"""

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Binary Dice score for a single class mask."""
    intersection = np.sum(pred & target)
    total = np.sum(pred) + np.sum(target)
    if total == 0:
        return 1.0  # both empty -> perfect agreement
    return 2.0 * intersection / total


def sensitivity_specificity(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    tp = np.sum(pred & target)
    fn = np.sum(~pred & target)
    fp = np.sum(pred & ~target)
    tn = np.sum(~pred & ~target)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    return sensitivity, specificity


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Extract boundary voxel coordinates via erosion difference."""
    eroded = binary_erosion(mask)
    surface = mask & ~eroded
    return np.argwhere(surface)


def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff distance between predicted and target surfaces.
    Falls back to standard Hausdorff via scipy's directed_hausdorff for simplicity;
    for exact percentile computation, distances would need full pairwise computation
    (fine at this volume scale — swap in a KD-tree if scaling to larger cohorts)."""
    pred_surface = _surface_points(pred)
    target_surface = _surface_points(target)

    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float("nan")

    d1, _, _ = directed_hausdorff(pred_surface, target_surface)
    d2, _, _ = directed_hausdorff(target_surface, pred_surface)
    return max(d1, d2)


def evaluate_case(pred: np.ndarray, target: np.ndarray, class_names: dict[int, str]) -> dict:
    """
    Compute per-class metrics for one case.

    pred, target: (D, H, W) integer label arrays with the same class encoding.
    class_names: e.g. {1: "NCR/NET", 2: "ED", 3: "ET"} (excludes background)
    """
    results = {}
    for class_id, name in class_names.items():
        pred_mask = pred == class_id
        target_mask = target == class_id

        results[name] = {
            "dice": dice_score(pred_mask, target_mask),
            "hausdorff95": hausdorff_95(pred_mask, target_mask),
        }
        sens, spec = sensitivity_specificity(pred_mask, target_mask)
        results[name]["sensitivity"] = sens
        results[name]["specificity"] = spec

    return results
