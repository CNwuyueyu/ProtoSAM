#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.dataloader_octa import OCTAInvertedDataset, build_octa_dataloader
from src.models.loss import soft_cldice, soft_dice_cldice
from src.models.model import ProtoFDA_SAM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage-2 OCTA few-shot training")

    # base
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--keep_ratio", action="store_true")

    # data
    parser.add_argument("--octa_name", type=str, default="", choices=["OCTA-6M", "OCTA-3M", "ROSE-1", "ROSE-O"])
    parser.add_argument("--octa_root", type=str, default="dataset/ROSE-1")
    parser.add_argument("--octa_json_path", type=str, default="json/ROSE-1_split.json")
    parser.add_argument("--k_shot", type=int, choices=[1, 3, 5], default=5)  #choose K
    parser.add_argument(
        "--support_key",
        type=str,
        default="",
    )

    # improvement
    parser.add_argument("--invert", action="store_true", default=True)
    parser.add_argument("--disable_invert", action="store_true")
    parser.add_argument("--enable_aug", action="store_true", default=True)
    parser.add_argument("--disable_aug", action="store_true")
    parser.add_argument(
        "--protocol",
        type=str,
        default="support_aug",
        choices=["strict", "support_aug"],
    )
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--self_cutmix_prob", type=float, default=1.0)

    # optimize
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_steps_per_epoch", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--max_val_steps", type=int, default=500)

    # loss
    parser.add_argument("--cldice_weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--use_official_soft_dice_cldice", action="store_true")

    # weights and freezing
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--pretrained_key", type=str, default="")
    parser.add_argument("--load_strict", action="store_true")
    parser.add_argument(
        "--freeze_mode",
        type=str,
        default="adapter_prompt",
        choices=["adapter_prompt", "all_trainable"],
        help="all_trainable ; adapter_prompt: train _adapter or prompt_gen ",
    )

    # output
    parser.add_argument("--output_dir", type=str, default="output_octa_stage2_new")

    return parser.parse_args()


def _extract_state_dict(ckpt_obj, preferred_key: str = ""):
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj

    if preferred_key:
        if preferred_key in ckpt_obj:
            return ckpt_obj[preferred_key]
        raise KeyError(f"preferred pretrained_key='{preferred_key}' not found in checkpoint")

    for k in ["model", "model_state_dict", "state_dict", "net", "network"]:
        if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
            return ckpt_obj[k]

    return ckpt_obj


def load_pretrained_weights(model: torch.nn.Module, ckpt_path: str, strict: bool = False, key: str = "") -> None:
    if not ckpt_path:
        print("[Info] No pretrained checkpoint provided, training from scratch.")
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt, preferred_key=key)
    incompatible = model.load_state_dict(state_dict, strict=strict)

    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    print(f"[Info] Loaded pretrained: {ckpt_path}")
    print(f"[Info] strict={strict}, missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")


def apply_freeze_policy(model: torch.nn.Module, freeze_mode: str) -> None:
    if freeze_mode == "all_trainable":
        for p in model.parameters():
            p.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        print(f"[Info] Freeze mode: all_trainable, trainable={total:,}")
        return

    trainable_keywords = ("_adapter", "prompt_gen")
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if any(k in name for k in trainable_keywords):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    print(f"[Info] Freeze mode: adapter_prompt, trainable_keywords={trainable_keywords}")
    print(f"[Info] frozen={frozen:,}, trainable={trainable:,}")


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def base_seg_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_with_logits(logits, targets)
    return 0.5 * bce + 0.5 * dice


def compute_total_loss(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
    cldice_fn: torch.nn.Module,
    cldice_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(pred_logits)
    base_loss = base_seg_loss(pred_logits, target_mask)
    cldice_loss = cldice_fn(target_mask, probs)
    total_loss = base_loss + cldice_weight * cldice_loss
    return total_loss, base_loss, cldice_loss


def compute_batch_dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * targets).flatten(1).sum(dim=1)
    denom = pred.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return float(dice.mean().item())


def compute_batch_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * targets).flatten(1).sum(dim=1)
    union = pred.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1) - inter
    iou = (inter + smooth) / (union + smooth)
    return float(iou.mean().item())


def compute_batch_cldice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cldice_fn: torch.nn.Module,
) -> float:
    probs = torch.sigmoid(logits)
    cldice_loss = cldice_fn(targets, probs)
    cldice_metric = 1.0 - cldice_loss
    return float(cldice_metric.item())


def read_support_names_from_json(json_path: str, k_shot: int, support_key: str = "") -> List[str]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"split json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    key = support_key if support_key else f"k{k_shot}"
    support_k = data.get("support_k", {})
    if key not in support_k:

        train_names = data.get("train", [])
        if len(train_names) == 0:
            raise KeyError(f"No support_k['{key}'] and train split empty in {json_path}")
        rng = random.Random(42)
        if len(train_names) >= k_shot:
            return sorted(rng.sample(train_names, k_shot))
        return sorted(rng.choices(train_names, k=k_shot))

    names = support_k[key]
    if len(names) == 0:
        raise RuntimeError(f"support_k['{key}'] is empty in {json_path}")
    return names


def filter_dataset_by_names(dataset: OCTAInvertedDataset, names: List[str]) -> None:
    wanted = set(names)
    kept = [s for s in dataset.samples if os.path.basename(s[0]) in wanted]
    if len(kept) == 0:
        raise RuntimeError(
            "No samples matched support names. "
            f"dataset={dataset.dataset_name}, split={dataset.split}, wanted={sorted(list(wanted))}"
        )
    dataset.samples = kept


def _dataset_image_paths(dataset: OCTAInvertedDataset) -> List[str]:
    return [os.path.abspath(s[0]) for s in dataset.samples]


def _support_image_paths(dataset: OCTAInvertedDataset, support_names: List[str]) -> List[str]:
    wanted = set(support_names)
    return [os.path.abspath(s[0]) for s in dataset.samples if os.path.basename(s[0]) in wanted]


def _log_split_overlap(
    train_paths: List[str],
    val_paths: List[str],
    test_paths: List[str],
    support_paths: List[str],
) -> None:
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)
    support_set = set(support_paths)

    overlap_train_val = len(train_set & val_set)
    overlap_train_test = len(train_set & test_set)
    overlap_val_test = len(val_set & test_set)
    overlap_support_val = len(support_set & val_set)
    overlap_support_test = len(support_set & test_set)

    print(
        "[Audit] overlap counts | "
        f"train∩val={overlap_train_val}, "
        f"train∩test={overlap_train_test}, "
        f"val∩test={overlap_val_test}, "
        f"support∩val={overlap_support_val}, "
        f"support∩test={overlap_support_test}"
    )

    if overlap_train_val or overlap_train_test or overlap_support_val or overlap_support_test:
        print("[Warn] Non-zero split overlap detected. Please verify your json split protocol.")


def build_fixed_support_tensors(
    dataset: OCTAInvertedDataset,
    support_names: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    support_images: (1, K, 3, H, W)
    support_masks:  (1, K, 1, H, W)
    """
    name_to_idx = {os.path.basename(s[0]): i for i, s in enumerate(dataset.samples)}

    support_imgs: List[torch.Tensor] = []
    support_msks: List[torch.Tensor] = []

    for n in support_names:
        if n not in name_to_idx:
            raise RuntimeError(f"Support sample '{n}' not found in dataset split '{dataset.split}'.")
        sample = dataset[name_to_idx[n]]
        support_imgs.append(sample["image"])
        support_msks.append(sample["mask"])

    support_images = torch.stack(support_imgs, dim=0).unsqueeze(0).to(device)  # (1,K,3,H,W)
    support_masks = torch.stack(support_msks, dim=0).unsqueeze(0).to(device)   # (1,K,1,H,W)
    return support_images, support_masks


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    support_images: torch.Tensor,
    support_masks: torch.Tensor,
    cldice_fn: torch.nn.Module,
    cldice_weight: float,
    device: torch.device,
    max_steps: int,
    desc: str,
) -> Dict[str, float]:
    model.eval()
    loss_sum, dice_sum, iou_sum, cldice_sum = 0.0, 0.0, 0.0, 0.0
    steps = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for i, batch in enumerate(pbar):
        if i >= max_steps:
            break

        q_img = batch["image"].to(device)
        q_mask = batch["mask"].to(device)

        outputs = model(
            q_img,
            support_images=support_images,
            support_masks=support_masks,
            training_stage="octa",
        )
        logits = outputs["masks"]

        loss, _, _ = compute_total_loss(logits, q_mask, cldice_fn, cldice_weight)
        dice = compute_batch_dice_from_logits(logits, q_mask)
        iou = compute_batch_iou_from_logits(logits, q_mask)
        cldice_metric = compute_batch_cldice_from_logits(logits, q_mask, cldice_fn)

        loss_sum += float(loss.item())
        dice_sum += dice
        iou_sum += iou
        cldice_sum += cldice_metric
        steps += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
                "iou": f"{iou:.4f}",
                "cldice": f"{cldice_metric:.4f}",
            }
        )

    if steps == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0, "cldice": 0.0}

    return {
        "loss": loss_sum / steps,
        "dice": dice_sum / steps,
        "iou": iou_sum / steps,
        "cldice": cldice_sum / steps,
    }


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, extra: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    ckpt.update(extra)
    torch.save(ckpt, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.disable_invert:
        args.invert = False
    if args.disable_aug:
        args.enable_aug = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device: {device}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.octa_name}_k{args.k_shot}_stage2"
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    # 1) support names from json
    support_names = read_support_names_from_json(
        json_path=args.octa_json_path,
        k_shot=args.k_shot,
        support_key=args.support_key,
    )
    print(f"[Info] support names ({len(support_names)}): {support_names}")

    # 2) loader

    train_enable_aug = args.enable_aug and args.protocol == "support_aug"
    train_cutmix_prob = args.cutmix_prob if train_enable_aug else 0.0
    train_self_cutmix_prob = args.self_cutmix_prob if train_enable_aug else 0.0

    print(f"[Info] protocol={args.protocol}, train_enable_aug={train_enable_aug}")

    train_loader = build_octa_dataloader(
        dataset_name=args.octa_name,
        dataset_root=args.octa_root,
        split="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        keep_ratio=args.keep_ratio,
        json_path=args.octa_json_path,
        return_names=False,
        seed=args.seed,
        invert=args.invert,
        enable_augmentation=train_enable_aug,
        cutmix_prob=train_cutmix_prob,
        self_cutmix_prob=train_self_cutmix_prob,
    )
    filter_dataset_by_names(train_loader.dataset, support_names)

    train_sample_count = len(train_loader.dataset)
    effective_batch_size = min(args.batch_size, train_sample_count)
    if effective_batch_size < 1:
        raise RuntimeError("Filtered train dataset is empty after applying support names.")

    if effective_batch_size != args.batch_size:
        print(
            f"[Warn] few-shot train samples={train_sample_count} < batch_size={args.batch_size}, "
            f"auto set train batch_size={effective_batch_size}"
        )
        #  DataLoader
        train_loader = build_octa_dataloader(
            dataset_name=args.octa_name,
            dataset_root=args.octa_root,
            split="train",
            img_size=args.img_size,
            batch_size=effective_batch_size,
            num_workers=args.num_workers,
            keep_ratio=args.keep_ratio,
            json_path=args.octa_json_path,
            return_names=False,
            seed=args.seed,
            invert=args.invert,
            enable_augmentation=train_enable_aug,
            cutmix_prob=train_cutmix_prob,
            self_cutmix_prob=train_self_cutmix_prob,
        )
        filter_dataset_by_names(train_loader.dataset, support_names)

    # 3) support_dataset
    support_dataset = OCTAInvertedDataset(
        dataset_name=args.octa_name,
        dataset_root=args.octa_root,
        split="train",
        img_size=args.img_size,
        keep_ratio=args.keep_ratio,
        json_path=args.octa_json_path,
        return_names=True,
        seed=args.seed,
        invert=args.invert,
        enable_augmentation=False,
    )
    support_images, support_masks = build_fixed_support_tensors(
        dataset=support_dataset,
        support_names=support_names,
        device=device,
    )

    # 4) val_loader
    val_loader = build_octa_dataloader(
        dataset_name=args.octa_name,
        dataset_root=args.octa_root,
        split="val",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        keep_ratio=args.keep_ratio,
        json_path=args.octa_json_path,
        return_names=False,
        seed=args.seed,
        invert=args.invert,
        enable_augmentation=False,
        cutmix_prob=0.0,
        self_cutmix_prob=0.0,
    )

    test_loader = build_octa_dataloader(
        dataset_name=args.octa_name,
        dataset_root=args.octa_root,
        split="test",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        keep_ratio=args.keep_ratio,
        json_path=args.octa_json_path,
        return_names=False,
        seed=args.seed,
        invert=args.invert,
        enable_augmentation=False,
        cutmix_prob=0.0,
        self_cutmix_prob=0.0,
    )

    train_paths = _dataset_image_paths(train_loader.dataset)
    val_paths = _dataset_image_paths(val_loader.dataset)
    test_paths = _dataset_image_paths(test_loader.dataset)
    support_paths = _support_image_paths(support_dataset, support_names)
    _log_split_overlap(train_paths, val_paths, test_paths, support_paths)

    # 5) optimizer, model, loss
    model = ProtoFDA_SAM().to(device)
    load_pretrained_weights(model, args.pretrained, strict=args.load_strict, key=args.pretrained_key)
    apply_freeze_policy(model, args.freeze_mode)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] model_total_params={total_params:,}, trainable_params={trainable_params_count:,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check freeze_mode and model module names.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    if args.use_official_soft_dice_cldice:
        cldice_fn = soft_dice_cldice(iter_=3, alpha=args.alpha).to(device)
        print(f"[Info] loss: BCE+Dice + soft_dice_cldice(alpha={args.alpha})")
    else:
        cldice_fn = soft_cldice(iter_=3).to(device)
        print("[Info] loss: BCE+Dice + soft_cldice")

    with open(os.path.join(save_dir, "support_names.json"), "w", encoding="utf-8") as f:
        json.dump({"support_names": support_names}, f, ensure_ascii=False, indent=2)

    # 6) training loop
    global_step = 0
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_val_cldice = -1.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_dice_sum = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs}")
        for step_idx, batch in enumerate(pbar):
            if step_idx >= args.max_steps_per_epoch:
                break

            adapt_img = batch["image"].to(device)
            adapt_mask = batch["mask"].to(device)

            outputs = model(
                adapt_img,
                support_images=support_images,
                support_masks=support_masks,
                training_stage="octa",
            )
            logits = outputs["masks"]

            total_loss, base_loss, cl_loss = compute_total_loss(
                pred_logits=logits,
                target_mask=adapt_mask,
                cldice_fn=cldice_fn,
                cldice_weight=args.cldice_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            dice = compute_batch_dice_from_logits(logits, adapt_mask)
            epoch_loss_sum += float(total_loss.item())
            epoch_dice_sum += dice
            steps += 1

            writer.add_scalar("train/total_loss_step", float(total_loss.item()), global_step)
            writer.add_scalar("train/base_loss_step", float(base_loss.item()), global_step)
            writer.add_scalar("train/cldice_loss_step", float(cl_loss.item()), global_step)
            writer.add_scalar("train/dice_step", dice, global_step)
            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "dice": f"{dice:.4f}",
                }
            )

        if steps == 0:
            raise RuntimeError("No training step executed. Check train loader and max_steps_per_epoch.")

        train_loss = epoch_loss_sum / steps
        train_dice = epoch_dice_sum / steps
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/dice_epoch", train_dice, epoch)

        # save last checkpoint
        save_checkpoint(
            path=os.path.join(save_dir, "last.pth"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            extra={
                "support_names": support_names,
                "args": vars(args),
                "best_val_dice": best_val_dice,
            },
        )

        # val and test evaluation every eval_interval epochs
        if (epoch + 1) % args.eval_interval == 0:
            val_metrics = evaluate_loader(
                model=model,
                loader=val_loader,
                support_images=support_images,
                support_masks=support_masks,
                cldice_fn=cldice_fn,
                cldice_weight=args.cldice_weight,
                device=device,
                max_steps=args.max_val_steps,
                desc="Validation",
            )
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/dice", val_metrics["dice"], epoch)
            writer.add_scalar("val/iou", val_metrics["iou"], epoch)
            writer.add_scalar("val/cldice", val_metrics["cldice"], epoch)

            print(
                f"[Epoch {epoch + 1}] "
                f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, val_dice={val_metrics['dice']:.4f}, "
                f"val_iou={val_metrics['iou']:.4f}, val_cldice={val_metrics['cldice']:.4f}"
            )

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                best_val_iou = val_metrics["iou"]
                best_val_cldice = val_metrics["cldice"]
                save_checkpoint(
                    path=os.path.join(save_dir, "best.pth"),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    extra={
                        "support_names": support_names,
                        "args": vars(args),
                        "best_val_dice": best_val_dice,
                        "best_val_iou": best_val_iou,
                        "best_val_cldice": best_val_cldice,
                    },
                )
                print(f"[Info] New best val dice: {best_val_dice:.4f}")

    # 7) test evaluation with best checkpoint
    best_path = os.path.join(save_dir, "best.pth")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=False)

    test_metrics = evaluate_loader(
        model=model,
        loader=test_loader,
        support_images=support_images,
        support_masks=support_masks,
        cldice_fn=cldice_fn,
        cldice_weight=args.cldice_weight,
        device=device,
        max_steps=10**9,
        desc="Test",
    )
    writer.add_scalar("test/loss", test_metrics["loss"], 0)
    writer.add_scalar("test/dice", test_metrics["dice"], 0)
    writer.add_scalar("test/iou", test_metrics["iou"], 0)
    writer.add_scalar("test/cldice", test_metrics["cldice"], 0)

    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"best_val_dice: {best_val_dice:.6f}\n")
        f.write(f"best_val_iou: {best_val_iou:.6f}\n")
        f.write(f"best_val_cldice: {best_val_cldice:.6f}\n")
        f.write(f"test_loss: {test_metrics['loss']:.6f}\n")
        f.write(f"test_dice: {test_metrics['dice']:.6f}\n")
        f.write(f"test_iou: {test_metrics['iou']:.6f}\n")
        f.write(f"test_cldice: {test_metrics['cldice']:.6f}\n")
        f.write(f"model_total_params: {total_params}\n")
        f.write(f"trainable_params: {trainable_params_count}\n")
        f.write(f"protocol: {args.protocol}\n")
        f.write(f"train_enable_aug: {train_enable_aug}\n")
        f.write(f"support_names: {support_names}\n")

    print(f"[Done] save_dir={save_dir}")
    print(
        f"[Done] best_val_dice={best_val_dice:.4f}, best_val_iou={best_val_iou:.4f}, "
        f"best_val_cldice={best_val_cldice:.4f}, test_dice={test_metrics['dice']:.4f}, "
        f"test_iou={test_metrics['iou']:.4f}, test_cldice={test_metrics['cldice']:.4f}"
    )

    writer.close()


if __name__ == "__main__":
    main()
