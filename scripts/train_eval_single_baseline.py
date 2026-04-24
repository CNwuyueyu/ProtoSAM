#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models.dataloader_octa import build_octa_dataloader
from src.models.backbones.FRUNet import FR_UNet
from src.models.backbones.Unet_part import DoubleConv, Down, OutConv, Up


DATASET_TO_ROOT = {
    "OCTA-3M": ROOT / "dataset" / "OCTA-3M",
    "OCTA-6M": ROOT / "dataset" / "OCTA-6M",
    "ROSE-1": ROOT / "dataset" / "ROSE-1",
}

DATASET_TO_JSON = {
    "OCTA-3M": ROOT / "json" / "OCTA-3M_split.json",
    "OCTA-6M": ROOT / "json" / "OCTA-6M_split.json",
    "ROSE-1": ROOT / "json" / "ROSE-1_split.json",
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


@dataclass
class BinaryMeter:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        # pred, gt: [B,1,H,W] in {0,1}
        p = pred.float()
        g = gt.float()
        self.tp += float((p * g).sum().item())
        self.fp += float((p * (1.0 - g)).sum().item())
        self.fn += float(((1.0 - p) * g).sum().item())

    def compute(self) -> Dict[str, float]:
        dice = (2.0 * self.tp + 1e-6) / (2.0 * self.tp + self.fp + self.fn + 1e-6)
        iou = (self.tp + 1e-6) / (self.tp + self.fp + self.fn + 1e-6)
        return {"dice": float(dice), "iou": float(iou)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def support_names_from_split(split_data: Dict, k: int, seed: int) -> List[str]:
    support_k = split_data.get("support_k", {})
    key = f"k{k}"
    if key in support_k:
        return list(support_k[key])

    train_names = list(split_data.get("train", []))
    if len(train_names) < k:
        raise ValueError(f"train split has {len(train_names)} samples, but k={k}")
    rng = random.Random(seed)
    return sorted(rng.sample(train_names, k))


def make_loader(
    dataset: str,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    invert: bool,
    enable_aug: bool,
    names_subset: Sequence[str] | None = None,
) -> DataLoader:
    root = str(DATASET_TO_ROOT[dataset])
    json_path = str(DATASET_TO_JSON[dataset])

    loader = build_octa_dataloader(
        dataset_name=dataset,
        dataset_root=root,
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        keep_ratio=False,
        json_path=json_path,
        return_names=True,
        seed=seed,
        invert=invert,
        enable_augmentation=enable_aug,
        cutmix_prob=0.5,
        self_cutmix_prob=1.0,
    )

    if names_subset is None:
        return loader

    wanted = set(os.path.basename(n) for n in names_subset)
    dataset_obj = loader.dataset
    kept = [s for s in dataset_obj.samples if os.path.basename(s[0]) in wanted]
    if not kept:
        raise RuntimeError(f"No samples matched the given subset in split={split}")
    dataset_obj.samples = kept

    eff_bs = min(batch_size, len(dataset_obj))
    return DataLoader(
        dataset_obj,
        batch_size=eff_bs,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train" and len(dataset_obj) > eff_bs),
    )


def sample_points_from_mask(mask: torch.Tensor, pos_num: int = 1, neg_num: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    # mask: [B,1,H,W] in {0,1}
    bsz, _, h, w = mask.shape
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


def _extract_state_dict(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj
    for k in ["model", "model_state_dict", "state_dict", "net", "network"]:
        if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
            return ckpt_obj[k]
    return ckpt_obj


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_sam_weights_flexible(sam_model: nn.Module, ckpt_path: str) -> None:
    """Load SAM backbone weights and auto-handle optional 'sam.' prefix."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state_dict = _extract_state_dict(ckpt)

    candidate_dicts = [raw_state_dict, _strip_prefix_if_present(raw_state_dict, "sam.")]
    seen = set()
    best = None
    best_score = -1

    for cand in candidate_dicts:
        key_sig = tuple(sorted(cand.keys())[:10])
        if key_sig in seen:
            continue
        seen.add(key_sig)

        incompatible = sam_model.load_state_dict(cand, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        matched = len(sam_model.state_dict()) - len(missing)
        score = matched - len(unexpected)

        if score > best_score:
            best_score = score
            best = (cand, missing, unexpected, matched)

    assert best is not None
    best_cand, best_missing, best_unexpected, best_matched = best
    sam_model.load_state_dict(best_cand, strict=False)
    print(
        "[Info] Loaded SAM init weights | "
        f"path={ckpt_path}, matched={best_matched}, "
        f"missing={len(best_missing)}, unexpected={len(best_unexpected)}"
    )


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
            sam = sam_model_registry["vit_b"](checkpoint=None)
            load_sam_weights_flexible(sam, args.sam_vit_b_ckpt)
            if args.sam_train_mode == "decoder_only":
                for p in sam.parameters():
                    p.requires_grad = False
                for p in sam.mask_decoder.parameters():
                    p.requires_grad = True
                for p in sam.prompt_encoder.parameters():
                    p.requires_grad = True
            else:
                for p in sam.parameters():
                    p.requires_grad = True
            model = VanillaSAMModel(sam, sam_resize=args.sam_resize).to(device)
            return model, "prob"

        # SAM-OCTA LoRA
        from sam_lora_image_encoder import LoRA_Sam  # type: ignore[import-not-found]

        sam = sam_model_registry["vit_b"](checkpoint=None)
        load_sam_weights_flexible(sam, args.sam_vit_b_ckpt)
        lora_model = LoRA_Sam(sam, args.lora_rank).to(device)
        return lora_model, "prob_with_prompt"

    if model_name == "learnable_sam":
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

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


def compute_loss_from_prob(prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    prob = torch.clamp(prob, 1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(prob, gt)
    inter = (prob * gt).flatten(1).sum(dim=1)
    denom = prob.flatten(1).sum(dim=1) + gt.flatten(1).sum(dim=1)
    dice_loss = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
    return 0.5 * bce + 0.5 * dice_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    model_type: str,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    training = mode == "train"
    model.train(training)

    meter = BinaryMeter()
    loss_sum = 0.0
    steps = 0

    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            if model_type == "logits":
                logits = model(image)
                prob = torch.sigmoid(logits)
            elif model_type == "prob":
                points, point_labels = sample_points_from_mask(mask, pos_num=1, neg_num=1)
                prob = model(image, points, point_labels)
            elif model_type == "prob_with_prompt":
                points, point_labels = sample_points_from_mask(mask, pos_num=1, neg_num=1)
                prob = model(image, tuple(image.shape[-2:]), points, point_labels)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            if prob.shape[-2:] != mask.shape[-2:]:
                prob = F.interpolate(prob, size=mask.shape[-2:], mode="bilinear", align_corners=False)

            loss = compute_loss_from_prob(prob, mask)

            if training:
                loss.backward()
                optimizer.step()

        pred_bin = (prob > 0.5).float()
        meter.update(pred_bin.detach(), mask.detach())
        loss_sum += float(loss.item())
        steps += 1

    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(steps, 1)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train/eval one baseline for one dataset and one k-shot")
    p.add_argument("--model", type=str, required=True, choices=["unet", "frunet", "sam", "learnable_sam", "sam_octa"])
    p.add_argument("--dataset", type=str, required=True, choices=["OCTA-3M", "OCTA-6M", "ROSE-1"])
    p.add_argument("--k", type=int, required=True, choices=[1, 3, 5])

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=str, default="0")

    p.add_argument("--invert", action="store_true", default=False)
    p.add_argument("--enable_aug", action="store_true", default=True)

    p.add_argument(
        "--sam_vit_b_ckpt",
        type=str,
        default=str(ROOT / "src" / "SAM-OCTA" / "sam_weights" / "sam_vit_b_01ec64.pth"),
    )
    p.add_argument("--sam_resize", type=int, default=1024)
    p.add_argument("--sam_train_mode", type=str, default="full", choices=["full", "decoder_only"])
    p.add_argument("--lora_rank", type=int, default=4)

    p.add_argument("--output_dir", type=str, default=str(ROOT / "output_baseline_compare"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.model in ["sam", "learnable_sam", "sam_octa"] and (not Path(args.sam_vit_b_ckpt).exists()):
        raise FileNotFoundError(
            f"SAM ViT-B checkpoint not found: {args.sam_vit_b_ckpt}. "
            "Please download sam_vit_b_01ec64.pth and set --sam_vit_b_ckpt."
        )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_data = read_split_json(DATASET_TO_JSON[args.dataset])
    support_names = support_names_from_split(split_data, args.k, args.seed)

    train_loader = make_loader(
        dataset=args.dataset,
        split="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        invert=args.invert,
        enable_aug=args.enable_aug,
        names_subset=support_names,
    )
    val_loader = make_loader(
        dataset=args.dataset,
        split="val",
        img_size=args.img_size,
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
        invert=args.invert,
        enable_aug=False,
        names_subset=None,
    )
    test_loader = make_loader(
        dataset=args.dataset,
        split="test",
        img_size=args.img_size,
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
        invert=args.invert,
        enable_aug=False,
        names_subset=None,
    )

    model, model_type = build_model(args, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / args.model / args.dataset / f"k{args.k}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    best_path = run_dir / "best.pth"
    history = []

    for ep in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, "train", model_type, optimizer)
        val_metrics = run_epoch(model, val_loader, device, "val", model_type, optimizer=None)

        record = {
            "epoch": ep,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        print(
            f"[Epoch {ep:03d}] "
            f"train_loss={train_metrics['loss']:.4f}, train_dice={train_metrics['dice']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_dice={val_metrics['dice']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = run_epoch(model, test_loader, device, "test", model_type, optimizer=None)

    final = {
        "model": args.model,
        "dataset": args.dataset,
        "k": args.k,
        "support_names": support_names,
        "best_val_dice": best_val_dice,
        "test": test_metrics,
        "history": history,
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print("[Done]", run_dir)
    print("[Test]", test_metrics)


if __name__ == "__main__":
    main()
