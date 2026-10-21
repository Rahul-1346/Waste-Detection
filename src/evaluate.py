import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from cnn import WasteCNN
except ImportError:
    from src.cnn import WasteCNN

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_DIR = REPO_ROOT / "archive" / "garbage-dataset-split" / "test"
DEFAULT_MODEL = REPO_ROOT / "models" / "best_waste_resnet18.pth"

def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        fallback = REPO_ROOT / checkpoint_path.name
        if fallback.exists():
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    is_cnn = "cnn" in checkpoint_path.name.lower() or ckpt.get("model_type") == "cnn"
    if is_cnn:
        model = WasteCNN(num_classes=len(class_names))
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names

def evaluate(checkpoint_path: Path, test_dir: Path, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not test_dir.exists():
        print(f"Error: Test directory not found at '{test_dir}'.")
        return

    model, class_names = load_model(checkpoint_path, device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(class_names)
    correct_per_class = {c: 0 for c in class_names}
    total_per_class = {c: 0 for c in class_names}
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            for pred, true in zip(predictions.cpu(), labels.cpu()):
                class_name = test_dataset.classes[true.item()]
                total_per_class[class_name] += 1
                if pred.item() == true.item():
                    correct_per_class[class_name] += 1

    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0

    print(f"OVERALL TEST ACCURACY: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
    print(f"{'Class Name':<16} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")

    for c_name in class_names:
        c_correct = correct_per_class.get(c_name, 0)
        c_total = total_per_class.get(c_name, 0)
        c_acc = (c_correct / c_total * 100) if c_total > 0 else 0.0
        print(f"{c_name:<16} {c_correct:<10} {c_total:<10} {c_acc:6.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    evaluate(checkpoint_path=args.model, test_dir=args.test_dir, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
