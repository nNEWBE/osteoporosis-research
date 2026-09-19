"""
Dataset Auditor and Stratified Manifest Generator
Identifies bitwise duplicates (MD5 hash) and cross-class contradictory pairs.
Generates clean, non-leaking train/val/test CSV manifests (70/15/15 split).
"""

import os
import hashlib
import random
import csv
import argparse
from typing import Dict, List, Tuple

def compute_md5(file_path: str) -> str:
    """Compute MD5 checksum for a file in binary chunks."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_dataset(data_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    """
    Audit image files in Normal/ and Osteoporosis/ directories.
    Returns:
      - normal_hashes: {hash: [list_of_filepaths]}
      - osteo_hashes: {hash: [list_of_filepaths]}
      - contradictory_hashes: [hashes present in both classes]
    """
    normal_dir = os.path.join(data_dir, "Normal")
    osteo_dir = os.path.join(data_dir, "Osteoporosis")

    assert os.path.exists(normal_dir), f"Directory not found: {normal_dir}"
    assert os.path.exists(osteo_dir), f"Directory not found: {osteo_dir}"

    normal_hashes = {}
    for f in sorted(os.listdir(normal_dir)):
        fp = os.path.join(normal_dir, f)
        if os.path.isfile(fp):
            h = compute_md5(fp)
            normal_hashes.setdefault(h, []).append(fp)

    osteo_hashes = {}
    for f in sorted(os.listdir(osteo_dir)):
        fp = os.path.join(osteo_dir, f)
        if os.path.isfile(fp):
            h = compute_md5(fp)
            osteo_hashes.setdefault(h, []).append(fp)

    contradictory = sorted(list(set(normal_hashes.keys()) & set(osteo_hashes.keys())))

    return normal_hashes, osteo_hashes, contradictory

def generate_stratified_manifests(
    data_dir: str,
    output_dir: str = "./manifests",
    deduplicate: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, int]:
    """
    Generate train.csv, val.csv, test.csv with strict stratification and zero data leakage.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    normal_hashes, osteo_hashes, contradictory = audit_dataset(data_dir)

    print("=" * 70)
    print("DATASET INTEGRITY AUDIT REPORT")
    print("=" * 70)
    print(f"Normal Class:       {sum(len(v) for v in normal_hashes.values())} files across {len(normal_hashes)} unique hashes")
    print(f"Osteoporosis Class: {sum(len(v) for v in osteo_hashes.values())} files across {len(osteo_hashes)} unique hashes")
    print(f"Cross-Class Contradictory Hashes: {len(contradictory)} (flagged in both classes)")

    clean_normal_files = []
    clean_osteo_files = []

    if deduplicate:
        # Exclude contradictory hashes completely
        valid_normal_hashes = set(normal_hashes.keys()) - set(contradictory)
        valid_osteo_hashes = set(osteo_hashes.keys()) - set(contradictory)

        # Pick exactly 1 canonical file for each unique hash
        for h in sorted(valid_normal_hashes):
            clean_normal_files.append((normal_hashes[h][0], 0, "Normal"))
        for h in sorted(valid_osteo_hashes):
            clean_osteo_files.append((osteo_hashes[h][0], 1, "Osteoporosis"))

        print(f"Post-Deduplication: {len(clean_normal_files)} Normal, {len(clean_osteo_files)} Osteoporosis (Total: {len(clean_normal_files) + len(clean_osteo_files)})")
    else:
        # Raw mode (keeps duplicates)
        for h, files in normal_hashes.items():
            for f in files:
                clean_normal_files.append((f, 0, "Normal"))
        for h, files in osteo_hashes.items():
            for f in files:
                clean_osteo_files.append((f, 1, "Osteoporosis"))
        print(f"Raw Mode: {len(clean_normal_files)} Normal, {len(clean_osteo_files)} Osteoporosis (Total: {len(clean_normal_files) + len(clean_osteo_files)})")

    # Shuffle deterministically
    random.shuffle(clean_normal_files)
    random.shuffle(clean_osteo_files)

    # Split indices for Normal
    n_norm = len(clean_normal_files)
    n_train_norm = int(n_norm * train_ratio)
    n_val_norm = int(n_norm * val_ratio)
    train_norm = clean_normal_files[:n_train_norm]
    val_norm = clean_normal_files[n_train_norm:n_train_norm + n_val_norm]
    test_norm = clean_normal_files[n_train_norm + n_val_norm:]

    # Split indices for Osteo
    n_ost = len(clean_osteo_files)
    n_train_ost = int(n_ost * train_ratio)
    n_val_ost = int(n_ost * val_ratio)
    train_ost = clean_osteo_files[:n_train_ost]
    val_ost = clean_osteo_files[n_train_ost:n_train_ost + n_val_ost]
    test_ost = clean_osteo_files[n_train_ost + n_val_ost:]

    train_data = train_norm + train_ost
    val_data = val_norm + val_ost
    test_data = test_norm + test_ost

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    splits = {
        "train.csv": train_data,
        "val.csv": val_data,
        "test.csv": test_data
    }

    print("-" * 70)
    for fname, data in splits.items():
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filepath", "label", "class_name"])
            for fp, lbl, cname in data:
                # Store relative path for cross-platform portability
                rel_path = os.path.relpath(fp, start=os.path.dirname(output_dir) if output_dir != "." else ".")
                writer.writerow([rel_path.replace("\\", "/"), lbl, cname])
        
        n_pos = sum(1 for _, l, _ in data if l == 1)
        n_neg = sum(1 for _, l, _ in data if l == 0)
        print(f"Saved {fname:10s} -> Total: {len(data):4d} | Normal (0): {n_neg:4d} | Osteo (1): {n_pos:4d}")
    print("=" * 70)

    return {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and partition knee osteoporosis dataset")
    parser.add_argument("--data_dir", type=str, default="OS Collected Data", help="Path to OS Collected Data")
    parser.add_argument("--output_dir", type=str, default="manifests", help="Output directory for manifests")
    parser.add_argument("--raw", action="store_true", help="Do not deduplicate (keep raw duplicate files)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_stratified_manifests(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        deduplicate=not args.raw,
        seed=args.seed
    )
