#!/usr/bin/env python3

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


@dataclass
class SplitResult:
    train: List[str]
    val: List[str]
    test: List[str]


def _find_images(img_dir: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    return sorted(files)


def _pairable_names(img_dir: str, gt_dir: str) -> List[str]:

    gt_stems = set()
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(gt_dir, f"*{ext}")):
            gt_stems.add(os.path.splitext(os.path.basename(p))[0])

    names: List[str] = []
    for p in _find_images(img_dir):
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        if stem in gt_stems:
            names.append(name)

    if not names:
        raise RuntimeError(f"No pairable samples found between {img_dir} and {gt_dir}")
    return sorted(names)


def _split_train_val_test(
    names: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> SplitResult:
    if not names:
        raise ValueError("Empty names list for split.")

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Require 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1")

    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    n_train = max(1, min(n_train, n - 1))
    n_val = max(0, min(n_val, n - n_train - 1))

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    if len(val) == 0 and len(train) >= 2:
        val = [train[-1]]
        train = train[:-1]

    if len(test) == 0 and len(train) >= 2:
        test = [train[-1]]
        train = train[:-1]

    return SplitResult(train=train, val=val, test=test)


def _pick_support_sets(train_names: List[str], ks: List[int], seed: int) -> Dict[str, List[str]]:

    if not train_names:
        raise ValueError("train_names is empty, cannot sample support sets.")

    rng = random.Random(seed)
    out: Dict[str, List[str]] = {}

    for k in ks:
        key = f"k{k}"
        if len(train_names) >= k:
            out[key] = sorted(rng.sample(train_names, k))
        else:

            out[key] = sorted(rng.choices(train_names, k=k))
    return out


def _build_octa6m_split(dataset_root: str, seed: int, train_ratio: float, val_ratio: float) -> Dict:
    img_dir = os.path.join(dataset_root, "img")
    gt_dir = os.path.join(dataset_root, "gt")
    if not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"OCTA-6M dirs not found: img={img_dir}, gt={gt_dir}")

    names = _pairable_names(img_dir, gt_dir)
    split = _split_train_val_test(names, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    support_k = _pick_support_sets(split.train, ks=[1, 3, 5], seed=seed + 101)

    return {
        "train": split.train,
        "val": split.val,
        "test": split.test,
        "support_k": support_k,
        "meta": {
            "dataset": "OCTA-6M",
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1.0 - train_ratio - val_ratio, 4),
            "counts": {
                "train": len(split.train),
                "val": len(split.val),
                "test": len(split.test),
            },
        },
    }


def _build_octa3m_split(dataset_root: str, seed: int, train_ratio: float, val_ratio: float) -> Dict:
    img_dir = os.path.join(dataset_root, "img")
    gt_dir = os.path.join(dataset_root, "gt")
    if not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"OCTA-3M dirs not found: img={img_dir}, gt={gt_dir}")

    names = _pairable_names(img_dir, gt_dir)
    split = _split_train_val_test(names, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    support_k = _pick_support_sets(split.train, ks=[1, 3, 5], seed=seed + 131)

    return {
        "train": split.train,
        "val": split.val,
        "test": split.test,
        "support_k": support_k,
        "meta": {
            "dataset": "OCTA-3M",
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1.0 - train_ratio - val_ratio, 4),
            "counts": {
                "train": len(split.train),
                "val": len(split.val),
                "test": len(split.test),
            },
        },
    }


def _build_rose_split(dataset_root: str, dataset_name: str, seed: int, val_ratio_within_train: float) -> Dict:
    """
    - train/img + train/gt 
    - test/img + test/gt 
    """
    train_img = os.path.join(dataset_root, "train", "img")
    train_gt = os.path.join(dataset_root, "train", "gt")
    test_img = os.path.join(dataset_root, "test", "img")
    test_gt = os.path.join(dataset_root, "test", "gt")

    for p in [train_img, train_gt, test_img, test_gt]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"{dataset_name} expected dir not found: {p}")

    train_all = _pairable_names(train_img, train_gt)
    test_names = _pairable_names(test_img, test_gt)

    if not (0.0 <= val_ratio_within_train < 1.0):
        raise ValueError("val_ratio_within_train must be in [0, 1).")

    rng = random.Random(seed)
    shuffled = train_all[:]
    rng.shuffle(shuffled)

    n_val = int(round(len(shuffled) * val_ratio_within_train))
    n_val = max(1, min(n_val, len(shuffled) - 1)) if len(shuffled) >= 2 else 0

    val_names = sorted(shuffled[:n_val])
    train_names = sorted(shuffled[n_val:])

    support_k = _pick_support_sets(train_names if train_names else train_all, ks=[1, 3, 5], seed=seed + 101)

    return {
        "train": train_names,
        "val": val_names,
        "test": sorted(test_names),
        "support_k": support_k,
        "meta": {
            "dataset": dataset_name,
            "seed": seed,
            "val_ratio_within_train": val_ratio_within_train,
            "counts": {
                "train": len(train_names),
                "val": len(val_names),
                "test": len(test_names),
            },
            "note": "ROSE uses original test folder as fixed test split; val is split from train folder.",
        },
    }


def _write_json(obj: Dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed OCTA split json files with support sets.")
    parser.add_argument("--dataset_root", type=str, default="../dataset", help="Root path containing OCTA-6M/OCTA-3M/ROSE-1/ROSE-O")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for json files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--octa6m_train_ratio", type=float, default=0.7)
    parser.add_argument("--octa6m_val_ratio", type=float, default=0.1)
    parser.add_argument("--octa3m_train_ratio", type=float, default=0.7)
    parser.add_argument("--octa3m_val_ratio", type=float, default=0.1)
    parser.add_argument("--rose_val_ratio", type=float, default=0.15, help="Val ratio split from ROSE train folder")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    output_dir = os.path.abspath(args.output_dir)

    # OCTA-6M
    # octa6m_json = _build_octa6m_split(
    #     dataset_root=os.path.join(dataset_root, "OCTA-6M"),
    #     seed=args.seed,
    #     train_ratio=args.octa6m_train_ratio,
    #     val_ratio=args.octa6m_val_ratio,
    # )
    # _write_json(octa6m_json, os.path.join(output_dir, "OCTA-6M_split.json"))

    # OCTA-3M
    octa3m_json = _build_octa3m_split(
        dataset_root=os.path.join(dataset_root, "OCTA-3M"),
        seed=args.seed,
        train_ratio=args.octa3m_train_ratio,
        val_ratio=args.octa3m_val_ratio,
    )
    _write_json(octa3m_json, os.path.join(output_dir, "OCTA-3M_split.json"))

    # ROSE-1
    # rose1_json = _build_rose_split(
    #     dataset_root=os.path.join(dataset_root, "ROSE-1"),
    #     dataset_name="ROSE-1",
    #     seed=args.seed,
    #     val_ratio_within_train=args.rose_val_ratio,
    # )
    # _write_json(rose1_json, os.path.join(output_dir, "ROSE-1_split.json"))


    print("Done. Generated files:")
    # print(os.path.join(output_dir, "OCTA-6M_split.json"))
    print(os.path.join(output_dir, "OCTA-3M_split.json"))
    # print(os.path.join(output_dir, "ROSE-1_split.json"))


if __name__ == "__main__":
    main()
