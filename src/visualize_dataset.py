import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "archive" / "garbage-dataset-split"

def visualize(dataset_dir: Path, output_file: Path = None, num_images: int = 12):
    train_dir = dataset_dir / "train"
    if not train_dir.exists():
        print(f"Error: Train directory not found at '{train_dir}'.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)

    if len(dataset) == 0:
        print("No images found in dataset.")
        return

    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes_flat):
        if i < len(dataset) and i < num_images:
            image, label = dataset[i]
            img_np = image.permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.set_title(dataset.classes[label].capitalize(), fontsize=12, fontweight="bold", pad=6)
        ax.axis("off")

    plt.tight_layout()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"Saved to: {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-images", type=int, default=12)
    args = parser.parse_args()

    visualize(dataset_dir=args.dataset_dir, output_file=args.output, num_images=args.num_images)

if __name__ == "__main__":
    main()
