import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _read_green_channel(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2:
        return img
    return img[:, :, 1]


def _resize_or_pad(img: np.ndarray, size: int, keep_ratio: bool, is_mask: bool) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    if not keep_ratio:
        return cv2.resize(img, (size, size), interpolation=interp)

    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    canvas = np.zeros((size, size), dtype=img.dtype)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def _to_tensor_image(gray: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)


def _to_tensor_mask(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)


def _to_3ch(img_1ch: torch.Tensor) -> torch.Tensor:
    return img_1ch.repeat(3, 1, 1)


def _read_split_list(json_path: str, split: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    if split == "all":
        merged = []
        for k in ["train", "val", "test"]:
            merged.extend(split_data.get(k, []))
        if not merged:
            raise KeyError(f"No train/val/test entries found in {json_path}")
        return merged

    if split not in split_data:
        raise KeyError(f"Split '{split}' not found in {json_path}")
    return split_data[split]


class CHASEDB1CFPDataset(Dataset):
    """Read CHASEDB1 CFP green channel and masks using split json."""

    def __init__(
        self,
        dataset_root: str,
        json_path: str,
        split: str = "train",
        img_size: int = 1024,
        keep_ratio: bool = False,
        return_names: bool = False,
    ):
        self.dataset_root = dataset_root
        self.json_path = json_path
        self.split = split
        self.img_size = img_size
        self.keep_ratio = keep_ratio
        self.return_names = return_names

        self.img_dir = os.path.join(dataset_root, "images")
        self.mask_dir = os.path.join(dataset_root, "1stho")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask dir not found: {self.mask_dir}")
        if not os.path.isfile(self.json_path):
            raise FileNotFoundError(f"Split json not found: {self.json_path}")

        names = _read_split_list(self.json_path, split)
        self.samples: List[Tuple[str, str, str]] = []

        for n in names:
            base = os.path.splitext(n)[0]
            img_path = os.path.join(self.img_dir, n)
            mask_path = os.path.join(self.mask_dir, f"{base}_1stHO.png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append((img_path, mask_path, base))

        if not self.samples:
            raise RuntimeError(
                "No CHASEDB1 samples matched split json. "
                f"dataset_root={dataset_root}, split={split}, json_path={json_path}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path, stem = self.samples[idx]

        cfp = _read_green_channel(img_path)
        mask = _read_gray(mask_path)

        cfp = _resize_or_pad(cfp, self.img_size, self.keep_ratio, is_mask=False)
        mask = _resize_or_pad(mask, self.img_size, self.keep_ratio, is_mask=True)

        image_1ch = _to_tensor_image(cfp)

        out: Dict[str, torch.Tensor] = {
            "image_1ch": image_1ch,
            "image": _to_3ch(image_1ch),
            "mask": _to_tensor_mask(mask),
        }
        if self.return_names:
            out["name"] = stem
        return out


def build_cfp_dataloader(
    dataset_root: str,
    json_path: str,
    split: str = "train",
    img_size: int = 1024,
    batch_size: int = 4,
    num_workers: int = 0,
    keep_ratio: bool = False,
    return_names: bool = False,
) -> DataLoader:
    dataset = CHASEDB1CFPDataset(
        dataset_root=dataset_root,
        json_path=json_path,
        split=split,
        img_size=img_size,
        keep_ratio=keep_ratio,
        return_names=return_names,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


if __name__ == "__main__":
    print("dataloader_cfp.py is ready.")
