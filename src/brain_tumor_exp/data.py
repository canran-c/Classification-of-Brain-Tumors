from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MedicalPreprocessTransform:
    def __init__(self, apply_pipeline: bool = False) -> None:
        self.apply_pipeline = apply_pipeline

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.apply_pipeline:
            return image

        gray = np.array(image.convert("L"), dtype=np.uint8)
        denoised = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.uint8)
        edges = np.array(Image.fromarray(denoised).filter(ImageFilter.FIND_EDGES), dtype=np.uint8)

        threshold = int(denoised.mean())
        mask = (denoised > threshold).astype(np.uint8) * 255
        segmented = (denoised * (mask > 0)).astype(np.uint8)

        merged = np.stack([segmented, denoised, edges], axis=-1)
        return Image.fromarray(merged)


def build_transforms(image_size: int, use_preprocessing: bool, use_augmentation: bool) -> tuple[Callable, Callable]:
    preprocess = MedicalPreprocessTransform(apply_pipeline=use_preprocessing)

    train_ops = [
        preprocess,
        transforms.Resize((image_size, image_size)),
    ]
    if use_augmentation:
        train_ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
        ]
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    eval_ops = [
        preprocess,
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def create_dataloaders(
    dataset_dir: Path,
    image_size: int,
    use_preprocessing: bool,
    use_augmentation: bool,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dir = dataset_dir / "Training"
    test_dir = dataset_dir / "Testing"

    train_transform, eval_transform = build_transforms(
        image_size=image_size,
        use_preprocessing=use_preprocessing,
        use_augmentation=use_augmentation,
    )

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, train_dataset.classes
