from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "archive" / "garbage-dataset-split"

def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Dict[str, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform
    }

def get_dataloaders(
    dataset_dir: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder]]:
    root = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory '{root}' does not exist.")

    all_transforms = get_transforms(image_size=image_size)
    dataset_splits = {}
    dataloaders = {}

    for split in ["train", "validation", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue

        dataset = datasets.ImageFolder(
            root=str(split_dir),
            transform=all_transforms[split if split in all_transforms else "val"]
        )
        dataset_splits[split] = dataset

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers
        )

    return dataloaders, dataset_splits

if __name__ == "__main__":
    if DEFAULT_DATASET_DIR.exists():
        loaders, datasets_map = get_dataloaders(batch_size=32)
        train_ds = datasets_map.get("train")
        if train_ds:
            print("Classes:", train_ds.classes)
            print("Class to index:", train_ds.class_to_idx)
            for split_name, ds in datasets_map.items():
                print(f"{split_name.capitalize()} size:", len(ds))
            batch_imgs, batch_lbls = next(iter(loaders["train"]))
            print("Batch shape:", batch_imgs.shape)
            print("Labels shape:", batch_lbls.shape)
    else:
        print("Dataset directory not found at:", DEFAULT_DATASET_DIR)