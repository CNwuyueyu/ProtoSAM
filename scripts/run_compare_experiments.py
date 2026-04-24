#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SINGLE_SCRIPT = ROOT / "scripts" / "train_eval_single_baseline.py"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Single launcher for one model, one dataset, one k-shot run")
    p.add_argument(
        "--model",
        default="sam_octa",
        choices=["unet", "frunet", "sam", "learnable_sam", "sam_octa"],
    )
    p.add_argument(
        "--dataset",
        default="ROSE-1",
        choices=["OCTA-3M", "OCTA-6M", "ROSE-1"],
    )
    p.add_argument("--k", type=int, default=5, choices=[1, 3, 5])

    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=str, default="0")

    p.add_argument("--invert", action="store_true", default=True)
    p.add_argument("--disable_aug", action="store_true", default=False)

    p.add_argument(
        "--sam_vit_b_ckpt",
        type=str,
        default="weight/sam_vit_b_01ec64.pth",
    )
    p.add_argument("--sam_resize", type=int, default=1024)
    p.add_argument("--sam_train_mode", type=str, default="full", choices=["full", "decoder_only"])
    p.add_argument("--lora_rank", type=int, default=4)

    p.add_argument("--output_dir", type=str, default=str(ROOT / "output_baseline_compare"))
    p.add_argument("--dry_run", action="store_true", default=False)
    p.add_argument("--continue_on_error", action="store_true", default=True)
    return p.parse_args()


def build_cmd(args: argparse.Namespace, model: str, dataset: str, k: int) -> list[str]:
    cmd = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--model",
        model,
        "--dataset",
        dataset,
        "--k",
        str(k),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--img_size",
        str(args.img_size),
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--sam_vit_b_ckpt",
        str(args.sam_vit_b_ckpt),
        "--sam_resize",
        str(args.sam_resize),
        "--sam_train_mode",
        str(args.sam_train_mode),
        "--lora_rank",
        str(args.lora_rank),
        "--output_dir",
        str(args.output_dir),
    ]
    if args.invert:
        cmd.append("--invert")
    if not args.disable_aug:
        cmd.append("--enable_aug")
    return cmd


def main() -> None:
    args = parse_args()

    jobs = [(args.model, args.dataset, args.k)]

    print(f"Total jobs: {len(jobs)}")

    failed = []
    for idx, (model, dataset, k) in enumerate(jobs, start=1):
        cmd = build_cmd(args, model, dataset, k)
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        print(f"[{idx}/{len(jobs)}] {model} | {dataset} | k={k}")
        print(cmd_str)

        if args.dry_run:
            continue

        ret = subprocess.run(cmd, cwd=str(ROOT))
        if ret.returncode != 0:
            failed.append((model, dataset, k, ret.returncode))
            print(f"[Error] return_code={ret.returncode}")
            if not args.continue_on_error:
                break

    if failed:
        print("\nFailed jobs:")
        for model, dataset, k, code in failed:
            print(f"- {model} | {dataset} | k={k} | code={code}")
        raise SystemExit(1)

    print("All jobs finished.")


if __name__ == "__main__":
    main()
