"""
PyTorch Dataset and DataLoader Pipeline for Knee Osteoporosis Radiographs.
Supports CSV manifests, ImageNet normalization, and training augmentations.
"""

import os
import csv
from typing import Tuple, Optional, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class KneeXRayDataset(Dataset):
    """
    Dataset loader reading from a manifest CSV (filepath, label, class_name)
    or directly from a root directory with class subfolders.
    """
    def __init__(
        self,
        manifest_path: Optional[str] = None,
        data_root: str = ".",
        transform: Optional[Callable] = None
    ):
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Resolve filepath relative to data_root
                    fp = os.path.normpath(os.path.join(data_root, row["filepath"]))
                    label = int(row["label"])
                    self.samples.append((fp, label))
        else:
            # Fallback direct folder reading
            for label, class_name in enumerate(["Normal", "Osteoporosis"]):
                cdir = os.path.join(data_root, class_name)
                if os.path.exists(cdir):
                    for fname in sorted(os.listdir(cdir)):
                        fp = os.path.join(cdir, fname)
                        if os.path.isfile(fp):
                            self.samples.append((fp, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, img_path


def get_data_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform)
    Uses ImageNet standard mean and std for transfer learning compatibility.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, eval_transform


def create_dataloaders(
    manifest_dir: str = "./manifests",
    data_root: str = ".",
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    """
    train_tf, eval_tf = get_data_transforms()

    train_dataset = KneeXRayDataset(
        manifest_path=os.path.join(manifest_dir, "train.csv"),
        data_root=data_root,
        transform=train_tf
    )
    val_dataset = KneeXRayDataset(
        manifest_path=os.path.join(manifest_dir, "val.csv"),
        data_root=data_root,
        transform=eval_tf
    )
    test_dataset = KneeXRayDataset(
        manifest_path=os.path.join(manifest_dir, "test.csv"),
        data_root=data_root,
        transform=eval_tf
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
