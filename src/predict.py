import argparse
from pathlib import Path
import sys
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from cnn import WasteCNN
except ImportError:
    from src.cnn import WasteCNN

REPO_ROOT = Path(__file__).resolve().parent.parent

WASTE_GUIDELINES = {
    "battery": {
        "status": "[Hazardous / E-Waste]",
        "badge": "Hazardous",
        "advice": "Never dispose in regular household trash or standard curbside recycling! Drop off at designated e-waste drop-off kiosks or battery collection points to prevent heavy metal contamination and fire hazards."
    },
    "biological": {
        "status": "[Organic / Compostable]",
        "badge": "Compostable",
        "advice": "Place in your green organic/food waste bin or home compost pile. Ideal for biological decomposition into nutrient-rich organic soil."
    },
    "cardboard": {
        "status": "[Recyclable - Paper & Cardboard]",
        "badge": "Recyclable",
        "advice": "Flatten all boxes to save space. Remove excessive plastic packing tape. Ensure it is clean and free of wet grease/oil before placing in paper recycling."
    },
    "clothes": {
        "status": "[Reusable / Textile Recycling]",
        "badge": "Textile",
        "advice": "If still in wearable condition, donate to local charities or clothing banks. If unwearable, deposit in textile recycling bins to be shredded and repurposed."
    },
    "glass": {
        "status": "[100% Infinitely Recyclable]",
        "badge": "Recyclable",
        "advice": "Rinse out liquids and food residue. Remove non-glass caps/corks. Place in designated glass recycling bin (separate clear, green, and brown glass if required locally)."
    },
    "metal": {
        "status": "[Highly Recyclable - Aluminum/Tin]",
        "badge": "Recyclable",
        "advice": "Rinse cans and metal containers. Lightly crush aluminum cans to save space. Metal can be recycled indefinitely without loss of quality."
    },
    "paper": {
        "status": "[Recyclable - Clean Paper]",
        "badge": "Recyclable",
        "advice": "Ensure paper is dry and uncontaminated by grease or food. Place in blue/paper recycling bin. Shredded paper should be bagged according to local council rules."
    },
    "plastic": {
        "status": "[Recyclable - Check Resin Code]",
        "badge": "Recyclable",
        "advice": "Empty all liquids and rinse residue. Check local resin recycling codes (e.g., PET #1, HDPE #2). Replace cap and place in plastic/curbside recycling."
    },
    "shoes": {
        "status": "[Reusable / Footwear Recycling]",
        "badge": "Reuse/Recycle",
        "advice": "Tie shoes together by laces if donating to thrift or charity stores. Heavily worn shoes can be dropped at specialized sneaker/footwear recycling collection programs."
    },
    "trash": {
        "status": "[Non-Recyclable Landfill Residual]",
        "badge": "Residual Waste",
        "advice": "Items here cannot be recycled through standard facilities (e.g. mixed composites, heavily soiled items). Dispose in standard general household waste."
    }
}

def load_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names")

    is_cnn = "cnn" in model_path.name.lower() or checkpoint.get("model_type") == "cnn"

    if is_cnn:
        model = WasteCNN(num_classes=len(class_names))
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names

def classify_image(image_path: Path, model_path: Path, top_k: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, min(top_k, len(class_names)))

    predicted_class = class_names[top_indices[0].item()]
    confidence = top_probs[0].item() * 100
    guideline = WASTE_GUIDELINES.get(predicted_class.lower(), {})

    print(f"Image File      : {image_path}")
    print(f"Model Checkpoint: {model_path.name}")
    print(f"Device          : {device}")
    print(f"PREDICTED CLASS : {predicted_class.upper()}")
    print(f"CONFIDENCE      : {confidence:.2f}%")
    print(f"CATEGORY STATUS : {guideline.get('status', 'Unknown')}")
    print(f"DISPOSAL ADVICE : {guideline.get('advice', 'Check local recycling regulations.')}")
    print(f"Top {top_k} Probability Distribution:")
    for prob, idx in zip(top_probs, top_indices):
        c_name = class_names[idx.item()]
        bar_len = int(prob.item() * 25)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        print(f"  {c_name:<14} : {prob.item() * 100:6.2f}% [{bar}]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    model_arg = args.model
    if Path(model_arg).exists():
        model_path = Path(model_arg)
    elif model_arg.lower() in ["resnet18", "resnet"]:
        model_path = REPO_ROOT / "models" / "best_waste_resnet18.pth"
        if not model_path.exists():
            model_path = REPO_ROOT / "best_waste_resnet18.pth"
    elif model_arg.lower() == "cnn":
        model_path = REPO_ROOT / "models" / "best_waste_cnn.pth"
    else:
        model_path = Path(model_arg)

    classify_image(image_path=args.image, model_path=model_path, top_k=args.top_k)

if __name__ == "__main__":
    main()
