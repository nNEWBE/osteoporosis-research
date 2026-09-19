"""
Comprehensive Model Evaluation & Statistical Benchmarking Suite.
Computes Accuracy, Sensitivity, Specificity, Precision, F1-Score, ROC-AUC,
and 95% Bootstrap Confidence Intervals for ResNet-101, DenseNet-201, Simple Average,
and the proposed Fuzzy Logic Fusion.
"""

import os
import json
import argparse
from typing import Dict, Tuple, List, Optional
import numpy as np

from fuzzy_engine import FuzzyFusionEngine, calibrate_fuzzy_threshold

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculate TP, TN, FP, FN."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute ROC-AUC via rank-sum (Wilcoxon-Mann-Whitney statistic),
    pure NumPy implementation matching sklearn roc_auc_score.
    """
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = np.argsort(np.argsort(y_prob)) + 1
    rank_sum_pos = np.sum(ranks[pos_mask])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate full set of clinical metrics given true labels and predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    sensitivity = tp / (tp + fn + 1e-9)  # Recall
    specificity = tn / (tn + fp + 1e-9)
    precision = tp / (tp + fp + 1e-9)    # PPV
    npv = tn / (tn + fn + 1e-9)          # Negative Predictive Value
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-9)
    auc = compute_roc_auc(y_true, y_prob)

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% Confidence Intervals via non-parametric bootstrapping with replacement.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_metrics = {
        "accuracy": [], "sensitivity": [], "specificity": [],
        "precision": [], "f1_score": [], "roc_auc": []
    }

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        # Skip resample if only 1 class is present
        if len(np.unique(y_true[indices])) < 2:
            continue
        m = compute_metrics(y_true[indices], y_prob[indices], threshold)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    ci_dict = {}
    for k, values in boot_metrics.items():
        if len(values) > 0:
            low = float(np.percentile(values, 2.5))
            high = float(np.percentile(values, 97.5))
            ci_dict[k] = (low, high)
        else:
            ci_dict[k] = (0.0, 0.0)

    return ci_dict

def benchmark_models(
    y_test: np.ndarray,
    p_resnet_test: np.ndarray,
    p_dense_test: np.ndarray,
    p_resnet_val: Optional[np.ndarray] = None,
    p_dense_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    output_dir: str = "./results"
) -> Dict:
    """
    Runs multi-model comparison:
    1. Standalone ResNet-101
    2. Standalone DenseNet-201
    3. Simple Probability Average
    4. Weighted Probability Average
    5. Proposed Fuzzy Logic Fusion (Mamdani)
    """
    os.makedirs(output_dir, exist_ok=True)
    engine = FuzzyFusionEngine()

    # 1. Standalone probabilities
    p_res = np.asarray(p_resnet_test, dtype=np.float64)
    p_den = np.asarray(p_dense_test, dtype=np.float64)

    # 2. Simple Average
    p_avg = 0.5 * (p_res + p_den)

    # 3. Weighted Average (tuned on val if available, else 0.5/0.5)
    w_best = 0.50
    if p_resnet_val is not None and p_dense_val is not None and y_val is not None:
        best_val_auc = 0.0
        for w in np.linspace(0.1, 0.9, 17):
            p_val_w = w * p_resnet_val + (1.0 - w) * p_dense_val
            auc = compute_roc_auc(y_val, p_val_w)
            if auc > best_val_auc:
                best_val_auc = auc
                w_best = w
    p_weighted = w_best * p_res + (1.0 - w_best) * p_den

    # 4. Fuzzy Fusion
    fuzzy_thresh = 0.50
    if p_resnet_val is not None and p_dense_val is not None and y_val is not None:
        fuzzy_thresh, _ = calibrate_fuzzy_threshold(p_resnet_val, p_dense_val, y_val, engine)
    p_fuzzy = engine.predict_proba(p_res, p_den)

    models_dict = {
        "ResNet-101": (p_res, 0.50),
        "DenseNet-201": (p_den, 0.50),
        "Simple Average": (p_avg, 0.50),
        f"Weighted Avg (w={w_best:.2f})": (p_weighted, 0.50),
        f"Fuzzy Fusion (tau={fuzzy_thresh:.2f})": (p_fuzzy, fuzzy_thresh)
    }

    results = {}
    print("\n" + "=" * 110)
    print(f"{'Model / Fusion Strategy':<30} | {'Accuracy (95% CI)':<22} | {'Sensitivity':<15} | {'Specificity':<15} | {'ROC-AUC (95% CI)':<20}")
    print("=" * 110)

    for name, (prob, thresh) in models_dict.items():
        m = compute_metrics(y_test, prob, threshold=thresh)
        ci = compute_bootstrap_ci(y_test, prob, threshold=thresh, n_bootstraps=1000)

        results[name] = {"metrics": m, "ci_95": ci}

        acc_str = f"{m['accuracy']*100:5.2f}% [{ci['accuracy'][0]*100:4.1f}-{ci['accuracy'][1]*100:4.1f}]"
        sens_str = f"{m['sensitivity']*100:5.2f}% [{ci['sensitivity'][0]*100:4.1f}-{ci['sensitivity'][1]*100:4.1f}]"
        spec_str = f"{m['specificity']*100:5.2f}% [{ci['specificity'][0]*100:4.1f}-{ci['specificity'][1]*100:4.1f}]"
        auc_str = f"{m['roc_auc']:.4f} [{ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}]"

        print(f"{name:<30} | {acc_str:<22} | {sens_str:<15} | {spec_str:<15} | {auc_str:<20}")

    print("=" * 110)

    # Save results to JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics and 95% CIs to: {json_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Benchmark Osteoporosis Detection Models")
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    # Synthetic demo verification
    np.random.seed(42)
    y_test_mock = np.array([0]*55 + [1]*57)
    # Simulated high-quality model predictions
    p_res_mock = np.clip(y_test_mock * 0.85 + np.random.normal(0.08, 0.12, len(y_test_mock)), 0.0, 1.0)
    p_den_mock = np.clip(y_test_mock * 0.88 + np.random.normal(0.06, 0.11, len(y_test_mock)), 0.0, 1.0)

    print("Running evaluation suite demonstration with 1,000 bootstrap iterations...")
    benchmark_models(y_test_mock, p_res_mock, p_den_mock, output_dir=args.results_dir)
