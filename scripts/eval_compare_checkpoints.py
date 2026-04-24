#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models.dataloader_octa import OCTAInvertedDataset
from src.models.loss import soft_cldice
from src.models.backbones.FRUNet import FR_UNet
from src.models.backbones.Unet_part import DoubleConv, Down, OutConv, Up


DATASET_DEFAULTS = {
    "OCTA-3M": {
        "root": "dataset/OCTA-3M",
        "json": "json/OCTA-3M_split.json",
    },
    "OCTA-6M": {
        "root": "dataset/OCTA-6M",
        "json": "json/OCTA-6M_split.json",
    },
    "ROSE-1": {
        "root": "dataset/ROSE-1",
        "json": "json/ROSE-1_split.json",
    },
}


class UNetBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class VanillaSAMModel(nn.Module):
    def __init__(self, sam_model: nn.Module, sam_resize: int = 1024):
        super().__init__()
        self.sam = sam_model
        from segment_anything.utils.transforms import ResizeLongestSide  # type: ignore[import-not-found]

        self.transform = ResizeLongestSide(sam_resize)

    def forward(self, images: torch.Tensor, points: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        original_size = tuple(images.shape[-2:])
        x = self.transform.apply_image_torch(images)
        p = self.transform.apply_coords_torch(points, original_size)
        input_size = tuple(x.shape[-2:])

        x = self.sam.preprocess(x)
        img_emb = self.sam.image_encoder(x)
        sparse_emb, dense_emb = self.sam.prompt_encoder(
            points=(p, point_labels), boxes=None, masks=None
        )
        low_res, _ = self.sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        masks = self.sam.postprocess_masks(low_res, input_size, original_size)
        return torch.sigmoid(masks)


@dataclass
class EvalResult:
    model_label: str
    model_type: str
    checkpoint: str
    dataset: str
    split: str
    k: int
    num_images: int
    mean_dice: float
    mean_iou: float
    mean_cldice: float
    mean_time_ms: float
    vis_sample_name: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate one baseline checkpoint")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=str, default="0")

    p.add_argument(
        "--model",
        type=str,
        default="sam",
        choices=["unet", "frunet", "sam", "learnable_sam", "sam_octa"]
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="weight/sam_vit_b_01ec64.pth",
        help="",
    )
    p.add_argument("--label", type=str, default="", help="")
    p.add_argument("--checkpoint_key", type=str, default="")
    p.add_argument("--load_strict", action="store_true")

    p.add_argument(
        "--dataset",
        type=str,

        default="ROSE-1",

        choices=["OCTA-3M", "OCTA-6M", "ROSE-1"]
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])

    p.add_argument("--dataset_root", type=str, default="", help="")
    p.add_argument("--dataset_json", type=str, default="", help="")

    p.add_argument("--k", type=int, default=5, choices=[1, 3, 5], help="k-shot")
    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--keep_ratio", action="store_true")
    p.add_argument("--invert", action="store_true", default=True)
    p.add_argument("--disable_invert", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument(
        "--sam_vit_b_ckpt",
        type=str,
        default="weight/sam_vit_b_01ec64.pth"
    )
    p.add_argument("--sam_resize", type=int, default=1024)
    p.add_argument("--sam_train_mode", type=str, default="full", choices=["full", "decoder_only"])
    p.add_argument("--lora_rank", type=int, default=4)

    #10302.bmp  10098.bmp  01.tif

    p.add_argument("--vis_sample_name", type=str, default="01.tif")

    p.add_argument("--vis_sample_index", type=int, default=0)
    p.add_argument("--vis_deinvert", action="store_true", default=True)

    p.add_argument("--output_dir", type=str, default="res_compare_eval/k=5")
    return p.parse_args()


def _extract_state_dict(ckpt_obj, preferred_key: str = ""):
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj

    if preferred_key:
        if preferred_key in ckpt_obj:
            return ckpt_obj[preferred_key]
        raise KeyError(f"preferred checkpoint_key='{preferred_key}' not found in checkpoint")

    for k in ["model", "model_state_dict", "state_dict", "net", "network"]:
        if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
            return ckpt_obj[k]

    return ckpt_obj


def _add_prefix_if_absent(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {prefix + k: v for k, v in state_dict.items()}


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, strict: bool = False, key: str = "") -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt, preferred_key=key)

    # Auto-adapt key prefixes for wrapper/core mismatches, e.g. 'sam.'
    candidates = [
        state_dict,
        _strip_prefix_if_present(state_dict, "sam."),
        _add_prefix_if_absent(state_dict, "sam."),
    ]

    best_incompatible = None
    best_candidate = None
    best_score = -10**9
    seen_key_count = set()

    for cand in candidates:
        sig = (len(cand), next(iter(cand.keys())) if len(cand) > 0 else "")
        if sig in seen_key_count:
            continue
        seen_key_count.add(sig)

        incompatible = model.load_state_dict(cand, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        matched = len(model.state_dict()) - len(missing)
        score = matched - len(unexpected)

        if score > best_score:
            best_score = score
            best_candidate = cand
            best_incompatible = incompatible

    assert best_candidate is not None and best_incompatible is not None

    # Final load with requested strictness on the best-matching key variant.
    incompatible = model.load_state_dict(best_candidate, strict=strict)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])

    print(f"[Info] Loaded checkpoint: {ckpt_path}")
    print(f"[Info] strict={strict}, missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")


def sample_points_from_mask(mask: torch.Tensor, pos_num: int = 1, neg_num: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = mask.shape[0]
    coords_all = []
    labels_all = []

    for b in range(bsz):
        m = mask[b, 0]
        pos = torch.nonzero(m > 0.5, as_tuple=False)
        neg = torch.nonzero(m <= 0.5, as_tuple=False)

        c_list: List[List[float]] = []
        l_list: List[int] = []

        if pos.numel() > 0:
            idx = torch.randint(0, pos.shape[0], (max(1, pos_num),), device=pos.device)
            for p in pos[idx]:
                y, x = p.tolist()
                c_list.append([float(x), float(y)])
                l_list.append(1)

        if neg.numel() > 0 and neg_num > 0:
            idx = torch.randint(0, neg.shape[0], (neg_num,), device=neg.device)
            for p in neg[idx]:
                y, x = p.tolist()
                c_list.append([float(x), float(y)])
                l_list.append(0)

        if not c_list:
            c_list = [[-100.0, -100.0]]
            l_list = [0]

        coords_all.append(torch.tensor(c_list, dtype=torch.float32, device=mask.device))
        labels_all.append(torch.tensor(l_list, dtype=torch.int64, device=mask.device))

    max_n = max(x.shape[0] for x in coords_all)
    coords_padded = []
    labels_padded = []

    for c, l in zip(coords_all, labels_all):
        if c.shape[0] < max_n:
            pad_n = max_n - c.shape[0]
            c = torch.cat([c, c.new_full((pad_n, 2), -100.0)], dim=0)
            l = torch.cat([l, l.new_zeros((pad_n,), dtype=l.dtype)], dim=0)
        coords_padded.append(c)
        labels_padded.append(l)

    return torch.stack(coords_padded, dim=0), torch.stack(labels_padded, dim=0)


def build_model(args: argparse.Namespace, device: torch.device):
    model_name = args.model

    if model_name == "unet":
        model = UNetBaseline(in_channels=3, n_classes=1).to(device)
        return model, "logits"

    if model_name == "frunet":
        model = FR_UNet(num_classes=1, num_channels=3).to(device)
        return model, "logits"

    if model_name in ["sam", "sam_octa"]:
        sam_repo = SRC_DIR / "SAM-OCTA"
        if str(sam_repo) not in sys.path:
            sys.path.insert(0, str(sam_repo))

        from segment_anything import sam_model_registry  # type: ignore[import-not-found]

        if model_name == "sam":
            sam = sam_model_registry["vit_b"](checkpoint=args.sam_vit_b_ckpt)
            model = VanillaSAMModel(sam, sam_resize=args.sam_resize).to(device)
            return model, "prob"

        from sam_lora_image_encoder import LoRA_Sam  # type: ignore[import-not-found]

        sam = sam_model_registry["vit_b"](checkpoint=args.sam_vit_b_ckpt)
        lora_model = LoRA_Sam(sam, args.lora_rank).to(device)
        return lora_model, "prob_with_prompt"

    if model_name == "learnable_sam":
        from baseline.LearnablePromptSAM.learnerable_seg import PromptSAM

        model = PromptSAM(
            model_name="vit_b",
            checkpoint=args.sam_vit_b_ckpt,
            num_classes=1,
            reduction=4,
            upsample_times=2,
            groups=4,
        ).to(device)
        return model, "logits"

    raise ValueError(f"Unsupported model: {model_name}")


def forward_to_prob(
    model: nn.Module,
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    model_type: str,
) -> torch.Tensor:
    if model_type == "logits":
        logits = model(image)
        prob = torch.sigmoid(logits)
    elif model_type == "prob":
        points, point_labels = sample_points_from_mask(gt_mask, pos_num=1, neg_num=1)
        prob = model(image, points, point_labels)
    elif model_type == "prob_with_prompt":
        points, point_labels = sample_points_from_mask(gt_mask, pos_num=1, neg_num=1)
        prob = model(image, tuple(image.shape[-2:]), points, point_labels)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if prob.shape[-2:] != gt_mask.shape[-2:]:
        prob = torch.nn.functional.interpolate(prob, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)

    return prob


def calc_dice_iou(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Tuple[float, float]:
    pred = pred_bin.float().view(-1)
    gt = target.float().view(-1)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    dice = (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice.item()), float(iou.item())


def calc_cldice(pred_bin: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    cldice_loss_fn = soft_cldice(iter_=3, smooth=smooth)
    cldice_loss = cldice_loss_fn(target.float(), pred_bin.float())
    cldice_score = 1.0 - cldice_loss
    return float(cldice_score.item())


def pick_sample_index(dataset: OCTAInvertedDataset, sample_name: str, sample_index: int) -> int:
    if sample_name:
        for i, item in enumerate(dataset.samples):
            if os.path.basename(item[0]) == sample_name:
                return i
        raise RuntimeError(f"vis_sample_name not found in split '{dataset.split}': {sample_name}")

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"vis_sample_index out of range: {sample_index}, dataset size={len(dataset)}")
    return sample_index


def save_visuals(
    save_dir: str,
    image_gray: np.ndarray,
    gt_mask: np.ndarray,
    pred_bin: np.ndarray,
    overlay: np.ndarray,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    plt.imsave(os.path.join(save_dir, "original.png"), image_gray, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(os.path.join(save_dir, "ground_truth.png"), gt_mask, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(os.path.join(save_dir, "prediction.png"), pred_bin, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(os.path.join(save_dir, "overlay.png"), overlay)


def build_tp_fp_fn_overlay(
    base_gray: np.ndarray,
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
) -> np.ndarray:
    """
    Build an RGB overlay where:
    - TP (correct vessel): green
    - FP (false positive): red
    - FN (missed vessel): yellow
    """
    base = np.clip(base_gray.astype(np.float32), 0.0, 1.0)
    overlay = np.stack([base, base, base], axis=-1)

    pred_mask = pred_bin > 0.5
    gt_mask = gt_bin > 0.5

    tp = pred_mask & gt_mask
    fp = pred_mask & (~gt_mask)
    fn = (~pred_mask) & gt_mask

    # Softer palette looks better on grayscale OCT backgrounds.
    color_tp = np.array([0.12, 0.85, 0.42], dtype=np.float32)  # mint green
    color_fp = np.array([1.00, 0.38, 0.38], dtype=np.float32)  # coral red
    color_fn = np.array([1.00, 0.80, 0.22], dtype=np.float32)  # amber yellow

    alpha_tp = 0.55
    alpha_fp = 0.62
    alpha_fn = 0.60

    overlay[tp] = (1.0 - alpha_tp) * overlay[tp] + alpha_tp * color_tp
    overlay[fp] = (1.0 - alpha_fp) * overlay[fp] + alpha_fp * color_fp
    overlay[fn] = (1.0 - alpha_fn) * overlay[fn] + alpha_fn * color_fn

    return np.clip(overlay, 0.0, 1.0)


def save_overlay_legend(save_dir: str) -> str:
    legend_path = os.path.join(save_dir, "overlay_legend.png")

    fig, ax = plt.subplots(figsize=(6.2, 1.4))
    ax.axis("off")

    labels = ["TP Correct", "FP False Positive", "FN Missed"]
    colors = [
        (0.12, 0.85, 0.42),
        (1.00, 0.38, 0.38),
        (1.00, 0.80, 0.22),
    ]

    x0 = 0.03
    step = 0.32
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = x0 + i * step
        rect = plt.Rectangle((x, 0.35), 0.08, 0.30, color=color, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.10, 0.50, label, va="center", ha="left", fontsize=10, transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(legend_path, dpi=200)
    plt.close(fig)
    return legend_path


def dataset_paths_from_args(args: argparse.Namespace, dataset_name: str) -> Tuple[str, str]:
    if args.dataset_root and args.dataset_json:
        return args.dataset_root, args.dataset_json

    if dataset_name == "OCTA-3M":
        return DATASET_DEFAULTS["OCTA-3M"]["root"], DATASET_DEFAULTS["OCTA-3M"]["json"]
    if dataset_name == "OCTA-6M":
        return DATASET_DEFAULTS["OCTA-6M"]["root"], DATASET_DEFAULTS["OCTA-6M"]["json"]
    if dataset_name == "ROSE-1":
        return DATASET_DEFAULTS["ROSE-1"]["root"], DATASET_DEFAULTS["ROSE-1"]["json"]
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def evaluate_one(
    model: torch.nn.Module,
    dataset: OCTAInvertedDataset,
    device: torch.device,
    threshold: float,
    model_type: str,
) -> Tuple[float, float, float, float]:
    dice_sum = 0.0
    iou_sum = 0.0
    cldice_sum = 0.0
    time_sum_ms = 0.0

    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["mask"].unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        if device.type == "cuda":
            t0.record()
        else:
            start = datetime.now()

        with torch.no_grad():
            prob = forward_to_prob(model, image, target, model_type)
            pred_bin = (prob > threshold).float()

        if device.type == "cuda":
            t1.record()
            torch.cuda.synchronize()
            elapsed_ms = float(t0.elapsed_time(t1))
        else:
            elapsed_ms = float((datetime.now() - start).total_seconds() * 1000.0)

        dice, iou = calc_dice_iou(pred_bin, target)
        cldice = calc_cldice(pred_bin, target)

        dice_sum += dice
        iou_sum += iou
        cldice_sum += cldice
        time_sum_ms += elapsed_ms

    n = max(1, len(dataset))
    return dice_sum / n, iou_sum / n, cldice_sum / n, time_sum_ms / n


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.disable_invert:
        args.invert = False

    if args.model in ["sam", "learnable_sam", "sam_octa"] and (not Path(args.sam_vit_b_ckpt).exists()):
        raise FileNotFoundError(
            f"SAM ViT-B checkpoint not found: {args.sam_vit_b_ckpt}. "
            "Please set --sam_vit_b_ckpt to the base SAM weight, e.g. weight/sam_vit_b_01ec64.pth."
        )

    if args.model in ["sam", "learnable_sam", "sam_octa"] and Path(args.sam_vit_b_ckpt).name == "best.pth":
        print(
            "[Warn] --sam_vit_b_ckpt seems to be a finetuned checkpoint (best.pth). "
            "Usually this argument should be the base SAM weight (sam_vit_b_01ec64.pth)."
        )

    label = args.label if args.label else Path(args.checkpoint).stem

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device: {device}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / f"compare_eval_{run_name}"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset
    dataset_root, dataset_json = dataset_paths_from_args(args, dataset_name)

    dataset = OCTAInvertedDataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        split=args.split,
        img_size=args.img_size,
        keep_ratio=args.keep_ratio,
        json_path=dataset_json,
        return_names=True,
        seed=args.seed,
        invert=args.invert,
        enable_augmentation=False,
    )

    vis_idx = pick_sample_index(dataset, args.vis_sample_name, args.vis_sample_index)
    vis_sample = dataset[vis_idx]
    vis_name = vis_sample.get("name", f"idx_{vis_idx:04d}")

    model, model_type = build_model(args, device)
    load_checkpoint(model, args.checkpoint, strict=args.load_strict, key=args.checkpoint_key)
    model.eval()

    mean_dice, mean_iou, mean_cldice, mean_time_ms = evaluate_one(
        model=model,
        dataset=dataset,
        device=device,
        threshold=args.threshold,
        model_type=model_type,
    )

    image = vis_sample["image"].unsqueeze(0).to(device)
    target = vis_sample["mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = forward_to_prob(model, image, target, model_type)
    pred_bin = (prob > args.threshold).float()

    image_np = vis_sample["image_1ch"].squeeze().cpu().numpy().astype(np.float32)
    if args.vis_deinvert and args.invert:
        vis_image_np = 1.0 - image_np
    else:
        vis_image_np = image_np
    gt_np = vis_sample["mask"].squeeze().cpu().numpy().astype(np.float32)
    pred_np = pred_bin.squeeze().cpu().numpy().astype(np.float32)


    overlay = build_tp_fp_fn_overlay(vis_image_np, pred_np, gt_np)

    vis_dir = out_root / "visualizations" / dataset_name / label
    save_visuals(str(vis_dir), vis_image_np, gt_np, pred_np, overlay)
    legend_path = save_overlay_legend(str(vis_dir))

    result = EvalResult(
        model_label=label,
        model_type=args.model,
        checkpoint=str(args.checkpoint),
        dataset=dataset_name,
        split=args.split,
        k=args.k,
        num_images=len(dataset),
        mean_dice=mean_dice,
        mean_iou=mean_iou,
        mean_cldice=mean_cldice,
        mean_time_ms=mean_time_ms,
        vis_sample_name=str(vis_name),
    )

    print(
        f"[Result] model={label} ({args.model}) | dataset={dataset_name} | "
        f"Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, clDice={mean_cldice:.4f}, "
        f"time={mean_time_ms:.3f} ms/img"
    )

    summary_csv = out_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_label",
                "model_type",
                "checkpoint",
                "dataset",
                "split",
                "k",
                "num_images",
                "mean_dice",
                "mean_iou",
                "mean_cldice",
                "mean_time_ms_per_image",
                "vis_sample_name",
            ]
        )
        writer.writerow(
            [
                result.model_label,
                result.model_type,
                result.checkpoint,
                result.dataset,
                result.split,
                result.k,
                result.num_images,
                f"{result.mean_dice:.6f}",
                f"{result.mean_iou:.6f}",
                f"{result.mean_cldice:.6f}",
                f"{result.mean_time_ms:.6f}",
                result.vis_sample_name,
            ]
        )

    summary_json = out_root / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, ensure_ascii=False, indent=2)

    print(f"[Done] summary_csv={summary_csv}")
    print(f"[Done] summary_json={summary_json}")
    print(f"[Done] legend={legend_path}")
    print(f"[Done] visualization_dir={out_root / 'visualizations'}")


if __name__ == "__main__":
    main()
