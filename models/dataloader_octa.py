import glob
import json
import os
import random
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


def _pair_by_stem(img_dir: str, mask_dir: str, names: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    img_exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]

    # mask  stem 
    mask_map = {}
    for ext in img_exts:
        for p in glob.glob(os.path.join(mask_dir, ext)):
            stem = os.path.splitext(os.path.basename(p))[0]
            mask_map[stem] = p

    if names is None:
        img_files: List[str] = []
        for ext in img_exts:
            img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    else:
        img_files = []
        for n in names:
            base = os.path.splitext(n)[0]
            found = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                p = os.path.join(img_dir, base + ext)
                if os.path.exists(p):
                    found = p
                    break
            if found is not None:
                img_files.append(found)

    pairs: List[Tuple[str, str, str]] = []
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if stem in mask_map:
            pairs.append((img_path, mask_map[stem], stem))

    pairs.sort(key=lambda x: x[2])
    if not pairs:
        raise RuntimeError(f"No paired samples found between {img_dir} and {mask_dir}")
    return pairs


# -------------------------
# augmentation
# -------------------------
def _random_rescaled_crop_and_resize(
    image: np.ndarray,
    mask: np.ndarray,
    out_size: int,
    rng: random.Random,
    scale_min: float = 0.6,
    scale_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    
    h, w = image.shape[:2]
    area = h * w

    for _ in range(10):
        target_area = area * rng.uniform(scale_min, scale_max)
        ratio = rng.uniform(0.75, 1.333)

        crop_w = int(round(np.sqrt(target_area * ratio)))
        crop_h = int(round(np.sqrt(target_area / ratio)))

        if 1 <= crop_h <= h and 1 <= crop_w <= w:
            y1 = rng.randint(0, h - crop_h)
            x1 = rng.randint(0, w - crop_w)
            img_crop = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
            msk_crop = mask[y1 : y1 + crop_h, x1 : x1 + crop_w]
            img_out = cv2.resize(img_crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            msk_out = cv2.resize(msk_crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
            return img_out, msk_out

    # 回退：不做裁剪，仅 resize
    img_out = cv2.resize(image, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    msk_out = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return img_out, msk_out


def _add_gaussian_noise(image: np.ndarray, rng: random.Random, sigma_min: float = 3.0, sigma_max: float = 15.0) -> np.ndarray:
    sigma = rng.uniform(sigma_min, sigma_max)
    noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_gamma(image: np.ndarray, rng: random.Random, gamma_min: float = 0.8, gamma_max: float = 1.2) -> np.ndarray:

    gamma = rng.uniform(gamma_min, gamma_max)
    x = image.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def _gaussian_or_motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        k = rng.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (k, k), sigmaX=0)

    k = rng.choice([3, 5, 7])
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= kernel.sum()
    out = cv2.filter2D(image, -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def _random_cutout(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:

    h, w = image.shape[:2]
    cut_w = rng.randint(max(1, w // 16), max(2, w // 6))
    cut_h = rng.randint(max(1, h // 16), max(2, h // 6))
    x1 = rng.randint(0, max(0, w - cut_w))
    y1 = rng.randint(0, max(0, h - cut_h))

    out = image.copy()
    fill = rng.randint(0, 255)
    out[y1 : y1 + cut_h, x1 : x1 + cut_w] = fill
    return out, mask


def _random_rotate(image: np.ndarray, mask: np.ndarray, rng: random.Random, max_deg: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    angle = rng.uniform(-max_deg, max_deg)
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)

    img_out = cv2.warpAffine(
        image,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    msk_out = cv2.warpAffine(
        mask,
        m,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return img_out, msk_out


# -------------------------
# -------------------------
class OCTAInvertedDataset(Dataset):

    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        split: str = "train",
        img_size: int = 1024,
        keep_ratio: bool = False,
        json_path: Optional[str] = None,
        image_subdir: Optional[str] = None,
        mask_subdir: Optional[str] = None,
        return_names: bool = False,
        seed: int = 42,
        invert: bool = True,
        enable_augmentation: bool = True,
        cutmix_prob: float = 0.5,
        self_cutmix_prob: float = 1.0,
    ):
        self.dataset_name = dataset_name.upper()
        self.dataset_root = dataset_root
        self.split = split
        self.img_size = img_size
        self.keep_ratio = keep_ratio
        self.return_names = return_names
        self.rng = random.Random(seed)

        self.invert = invert
        self.enable_augmentation = enable_augmentation and (split == "train")
        self.cutmix_prob = cutmix_prob
        self.self_cutmix_prob = self_cutmix_prob

        if self.dataset_name in ["OCTA-6M", "OCTA6M", "OCTA-3M", "OCTA3M", "OCTA-500", "OCTA500"]:
            image_dir = os.path.join(dataset_root, "img") if image_subdir is None else os.path.join(dataset_root, image_subdir)
            mask_dir = os.path.join(dataset_root, "gt") if mask_subdir is None else os.path.join(dataset_root, mask_subdir)
            if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
                raise FileNotFoundError(
                    f"OCTA expected dirs not found. image_dir={image_dir}, mask_dir={mask_dir}"
                )

        elif self.dataset_name in ["ROSE", "ROSE-1", "ROSE1", "ROSE-O", "ROSEO"]:
            # ROSE-1
            if json_path and os.path.exists(json_path):
                split_folder = "test" if split == "test" else "train"
            else:
                split_folder = "test" if split in ["test", "val"] else "train"
            if image_subdir is None:
                image_dir = os.path.join(dataset_root, split_folder, "img")
            else:
                image_dir = os.path.join(dataset_root, image_subdir)

            if mask_subdir is None:
                mask_dir = os.path.join(dataset_root, split_folder, "gt")
            else:
                mask_dir = os.path.join(dataset_root, mask_subdir)

            if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
                raise FileNotFoundError(
                    f"ROSE expected dirs not found. image_dir={image_dir}, mask_dir={mask_dir}"
                )
        else:
            raise ValueError(f"Unsupported OCTA dataset_name: {dataset_name}")

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        names = _read_split_list(json_path, split) if (json_path and os.path.exists(json_path)) else None
        self.samples = _pair_by_stem(self.image_dir, self.mask_dir, names)
        if not self.samples:
            raise RuntimeError("No OCTA samples found.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path, mask_path, _ = self.samples[idx]
        image = _read_gray(img_path)
        mask = _read_gray(mask_path)

        image = _resize_or_pad(image, self.img_size, self.keep_ratio, is_mask=False)
        mask = _resize_or_pad(mask, self.img_size, self.keep_ratio, is_mask=True)

        if self.invert:
            image = 255 - image

        return image, mask

    def _self_region_mix(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        h, w = image.shape[:2]
        beta = 1.0
        lam = np.random.beta(beta, beta)
        cut_ratio = np.sqrt(1.0 - lam)

        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cut_w = max(1, min(cut_w, w - 1))
        cut_h = max(1, min(cut_h, h - 1))

        sx = self.rng.randint(0, w - cut_w)
        sy = self.rng.randint(0, h - cut_h)

        dx = self.rng.randint(0, w - cut_w)
        dy = self.rng.randint(0, h - cut_h)

        mixed_img = image.copy()
        mixed_msk = mask.copy()
        mixed_img[dy : dy + cut_h, dx : dx + cut_w] = image[sy : sy + cut_h, sx : sx + cut_w]
        mixed_msk[dy : dy + cut_h, dx : dx + cut_w] = mask[sy : sy + cut_h, sx : sx + cut_w]
        return mixed_img, mixed_msk

    def _apply_cutmix(self, image: np.ndarray, mask: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutMix
        """
        if len(self.samples) <= 1:
    
            if self.rng.random() < self.self_cutmix_prob:
                return self._self_region_mix(image, mask)
            return image, mask

        mix_idx = self.rng.randint(0, len(self.samples) - 1)
        while mix_idx == idx:
            mix_idx = self.rng.randint(0, len(self.samples) - 1)

        mix_image, mix_mask = self._load_raw_sample(mix_idx)

        h, w = image.shape[:2]
        beta = 1.0
        lam = np.random.beta(beta, beta)
        cut_ratio = np.sqrt(1.0 - lam)

        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cut_w = max(1, min(cut_w, w - 1))
        cut_h = max(1, min(cut_h, h - 1))

        cx = self.rng.randint(cut_w // 2, w - cut_w // 2)
        cy = self.rng.randint(cut_h // 2, h - cut_h // 2)

        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(w, cx + cut_w // 2)
        bby2 = min(h, cy + cut_h // 2)

        mixed_image = image.copy()
        mixed_mask = mask.copy()

        mixed_image[bby1:bby2, bbx1:bbx2] = mix_image[bby1:bby2, bbx1:bbx2]
        mixed_mask[bby1:bby2, bbx1:bbx2] = mix_mask[bby1:bby2, bbx1:bbx2]

        return mixed_image, mixed_mask

    def _augment_train(self, image: np.ndarray, mask: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:

        # 1) random crop+resize
        if self.rng.random() < 0.8:
            image, mask = _random_rescaled_crop_and_resize(
                image=image,
                mask=mask,
                out_size=self.img_size,
                rng=self.rng,
                scale_min=0.6,
                scale_max=1.0,
            )

        # 2) random flip
        if self.rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        if self.rng.random() < 0.2:
            image = np.ascontiguousarray(image[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])

        # 3) rotation
        if self.rng.random() < 0.5:
            image, mask = _random_rotate(image, mask, self.rng, max_deg=20.0)

        # 4) motion
        if self.rng.random() < 0.5:
            image = _gaussian_or_motion_blur(image, self.rng)

        # 5) gaussian noise
        if self.rng.random() < 0.7:
            image = _add_gaussian_noise(image, self.rng, sigma_min=3.0, sigma_max=15.0)

        # 6) gamma
        if self.rng.random() < 0.5:
            image = _apply_gamma(image, self.rng, gamma_min=0.8, gamma_max=1.2)

        # 7) Cutout
        if self.rng.random() < 0.4:
            image, mask = _random_cutout(image, mask, self.rng)

        # 8) CutMix
        if self.rng.random() < self.cutmix_prob:
            image, mask = self._apply_cutmix(image, mask, idx)

        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, _, stem = self.samples[idx]
        image, mask = self._load_raw_sample(idx)

        if self.enable_augmentation:
            image, mask = self._augment_train(image, mask, idx)

        image_1ch = _to_tensor_image(image)
        out: Dict[str, torch.Tensor] = {
            "image_1ch": image_1ch,
            "image": _to_3ch(image_1ch),
            "mask": _to_tensor_mask(mask),
        }
        if self.return_names:
            out["name"] = stem
            out["image_path"] = img_path
        return out


# -------------------------
# DataLoader
# -------------------------
def build_octa_dataloader(
    dataset_name: str,
    dataset_root: str,
    split: str = "train",
    img_size: int = 1024,
    batch_size: int = 2,
    num_workers: int = 0,
    keep_ratio: bool = False,
    json_path: Optional[str] = None,
    image_subdir: Optional[str] = None,
    mask_subdir: Optional[str] = None,
    return_names: bool = False,
    seed: int = 42,
    invert: bool = True,
    enable_augmentation: bool = True,
    cutmix_prob: float = 0.5,
    self_cutmix_prob: float = 1.0,
) -> DataLoader:
    dataset = OCTAInvertedDataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        split=split,
        img_size=img_size,
        keep_ratio=keep_ratio,
        json_path=json_path,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        return_names=return_names,
        seed=seed,
        invert=invert,
        enable_augmentation=enable_augmentation,
        cutmix_prob=cutmix_prob,
        self_cutmix_prob=self_cutmix_prob,
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
    print("dataloader_octa.py is ready.")
