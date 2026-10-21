import argparse
from pathlib import Path
import random
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = REPO_ROOT / "archive" / "garbage-dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "archive" / "garbage-dataset-split"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_CLASSES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
):
    random.seed(seed)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    discovered = [d.name for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    classes = sorted(discovered) if discovered else DEFAULT_CLASSES

    total_images_processed = 0

    for class_name in classes:
        source_class_dir = source_dir / class_name
        if not source_class_dir.exists():
            continue

        images = [
            f for f in source_class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            continue

        random.shuffle(images)
        total = len(images)
        total_images_processed += total

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        train_dir = output_dir / "train" / class_name
        val_dir = output_dir / "validation" / class_name
        test_dir = output_dir / "test" / class_name

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, train_dir / img.name)
        for img in val_imgs:
            shutil.copy2(img, val_dir / img.name)
        for img in test_imgs:
            shutil.copy2(img, test_dir / img.name)

        print(
            f"{class_name:<14} Total: {total:<5} | "
            f"Train: {len(train_imgs):<5} | "
            f"Val: {len(val_imgs):<5} | "
            f"Test: {len(test_imgs):<5}"
        )

    print(f"Total images processed: {total_images_processed}")
    print(f"Dataset ready at: {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    split_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()