#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.dataloader_cfp import build_cfp_dataloader
from src.models.loss import soft_cldice, soft_dice_cldice
from src.models.model import ProtoFDA_SAM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage-1 CFP training")

    # base
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--keep_ratio", action="store_true")

    # data
    parser.add_argument("--cfp_root", type=str, default="dataset/CHASEDB1")
    parser.add_argument("--cfp_json_path", type=str, default="json/CHASEDB1_split.json")

    # optimize
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_steps_per_epoch", type=int, default=400)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--max_val_steps", type=int, default=200)

    # loss
    parser.add_argument("--cldice_weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--use_official_soft_dice_cldice", action="store_true")

    # weights and freezing
    parser.add_argument("--pretrained", type=str, default="weight/mobile_sam_fix.pth")
    parser.add_argument("--pretrained_key", type=str, default="")
    parser.add_argument("--load_strict", action="store_true")
    parser.add_argument(
        "--freeze_mode",
        type=str,
        default="adapter_prompt",
        choices=["all_trainable", "adapter_prompt"],
        help="all_trainable ; adapter_prompt: train _adapter or prompt_gen ",
    )

    # output
    parser.add_argument("--output_dir", type=str, default="output_cfp_stage1")

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


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    cldice_fn: torch.nn.Module,
    cldice_weight: float,
    device: torch.device,
    max_steps: int,
    desc: str,
) -> Dict[str, float]:
    model.eval()
    loss_sum, dice_sum = 0.0, 0.0
    steps = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for i, batch in enumerate(pbar):
        if i >= max_steps:
            break

        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        outputs = model(image, training_stage="cfp")
        logits = outputs["masks"]

        loss, _, _ = compute_total_loss(logits, mask, cldice_fn, cldice_weight)
        dice = compute_batch_dice_from_logits(logits, mask)

        loss_sum += float(loss.item())
        dice_sum += dice
        steps += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})

    if steps == 0:
        return {"loss": 0.0, "dice": 0.0}

    return {"loss": loss_sum / steps, "dice": dice_sum / steps}


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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device: {device}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_CHASEDB1_stage1"
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    train_loader = build_cfp_dataloader(
        dataset_root=args.cfp_root,
        json_path=args.cfp_json_path,
        split="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        keep_ratio=args.keep_ratio,
        return_names=False,
    )
    val_loader = build_cfp_dataloader(
        dataset_root=args.cfp_root,
        json_path=args.cfp_json_path,
        split="val",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        keep_ratio=args.keep_ratio,
        return_names=False,
    )
    test_loader = build_cfp_dataloader(
        dataset_root=args.cfp_root,
        json_path=args.cfp_json_path,
        split="test",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        keep_ratio=args.keep_ratio,
        return_names=False,
    )

    model = ProtoFDA_SAM().to(device)
    load_pretrained_weights(model, args.pretrained, strict=args.load_strict, key=args.pretrained_key)
    apply_freeze_policy(model, args.freeze_mode)

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

    global_step = 0
    best_val_dice = -1.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_dice_sum = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs}")
        for step_idx, batch in enumerate(pbar):
            if step_idx >= args.max_steps_per_epoch:
                break

            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(image, training_stage="cfp")
            logits = outputs["masks"]

            total_loss, base_loss, cl_loss = compute_total_loss(
                pred_logits=logits,
                target_mask=mask,
                cldice_fn=cldice_fn,
                cldice_weight=args.cldice_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            dice = compute_batch_dice_from_logits(logits, mask)
            epoch_loss_sum += float(total_loss.item())
            epoch_dice_sum += dice
            steps += 1

            writer.add_scalar("train/total_loss_step", float(total_loss.item()), global_step)
            writer.add_scalar("train/base_loss_step", float(base_loss.item()), global_step)
            writer.add_scalar("train/cldice_loss_step", float(cl_loss.item()), global_step)
            writer.add_scalar("train/dice_step", dice, global_step)
            global_step += 1

            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "dice": f"{dice:.4f}"})

        if steps == 0:
            raise RuntimeError("No training step executed. Check train loader and max_steps_per_epoch.")

        train_loss = epoch_loss_sum / steps
        train_dice = epoch_dice_sum / steps
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/dice_epoch", train_dice, epoch)

        save_checkpoint(
            path=os.path.join(save_dir, "last.pth"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            extra={
                "args": vars(args),
                "best_val_dice": best_val_dice,
            },
        )

        if (epoch + 1) % args.eval_interval == 0:
            val_metrics = evaluate_loader(
                model=model,
                loader=val_loader,
                cldice_fn=cldice_fn,
                cldice_weight=args.cldice_weight,
                device=device,
                max_steps=args.max_val_steps,
                desc="Validation",
            )
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/dice", val_metrics["dice"], epoch)

            print(
                f"[Epoch {epoch + 1}] "
                f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, val_dice={val_metrics['dice']:.4f}"
            )

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                save_checkpoint(
                    path=os.path.join(save_dir, "best.pth"),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    extra={
                        "args": vars(args),
                        "best_val_dice": best_val_dice,
                    },
                )
                print(f"[Info] New best val dice: {best_val_dice:.4f}")

    best_path = os.path.join(save_dir, "best.pth")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=False)

    test_metrics = evaluate_loader(
        model=model,
        loader=test_loader,
        cldice_fn=cldice_fn,
        cldice_weight=args.cldice_weight,
        device=device,
        max_steps=10**9,
        desc="Test",
    )

    writer.add_scalar("test/loss", test_metrics["loss"], 0)
    writer.add_scalar("test/dice", test_metrics["dice"], 0)

    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"best_val_dice: {best_val_dice:.6f}\n")
        f.write(f"test_loss: {test_metrics['loss']:.6f}\n")
        f.write(f"test_dice: {test_metrics['dice']:.6f}\n")

    print(f"[Done] save_dir={save_dir}")
    print(f"[Done] best_val_dice={best_val_dice:.4f}, test_dice={test_metrics['dice']:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
