import argparse
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models

try:
    from cnn import WasteCNN
    from dataset import get_dataloaders
except ImportError:
    from src.cnn import WasteCNN
    from src.dataset import get_dataloaders

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "archive" / "garbage-dataset-split"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models"

def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name.lower() == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.lower() == "cnn":
        model = WasteCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def train_model(
    model_name: str = "resnet18",
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    image_size: int = 128,
    num_workers: int = 0
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, datasets_map = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=(image_size, image_size)
    )

    if "train" not in dataloaders or "validation" not in dataloaders:
        raise RuntimeError("Dataset must contain both train and validation directories.")

    train_loader = dataloaders["train"]
    val_loader = dataloaders["validation"]

    class_names = datasets_map["train"].classes
    num_classes = len(class_names)

    model = build_model(model_name, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    checkpoint_name = f"best_waste_{model_name.lower()}.pth"
    save_path = output_dir / checkpoint_name

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "validation_accuracy": val_acc,
                "model_type": model_name.lower(),
                "image_size": (image_size, image_size),
                "epoch": epoch
            }, save_path)

    total_time = time.time() - start_time
    print(f"Training Complete in {total_time // 60:.0f}m {total_time % 60:.1f}s")
    print(f"Best Validation Accuracy: {best_val_acc:.2%}")
    print(f"Saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "cnn"])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    lr = args.lr
    if lr is None:
        lr = 1e-4 if args.model.lower() == "resnet18" else 1e-3

    train_model(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=lr,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
