from pathlib import Path
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(
    page_title="EcoSort AI | Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

WASTE_DATA = {
    "battery": {
        "icon": "🔋",
        "badge": "Hazardous E-Waste",
        "badge_color": "#e63946",
        "bin": "E-Waste / Battery Drop-Off",
        "recyclable": False,
        "advice": (
            "Never throw in household trash or standard curbside recycling! "
            "Batteries contain toxic heavy metals that cause landfill fires. "
            "Deposit in dedicated battery recycling bins at local supermarkets or civic waste centers."
        )
    },
    "biological": {
        "icon": "🍎",
        "badge": "Organic / Compostable",
        "badge_color": "#2a9d8f",
        "bin": "Green Organic / Compost Bin",
        "recyclable": True,
        "advice": (
            "Biodegradable organic matter. Place in your green organics bin or home compost pile. "
            "Converts organic waste into nutrient-rich compost soil."
        )
    },
    "cardboard": {
        "icon": "📦",
        "badge": "Recyclable (Paper & Board)",
        "badge_color": "#457b9d",
        "bin": "Blue Paper/Cardboard Bin",
        "recyclable": True,
        "advice": (
            "Flatten boxes to maximize bin space. Remove plastic packaging and heavy packing tape. "
            "Ensure cardboard is dry and free of pizza grease or food contamination before recycling."
        )
    },
    "clothes": {
        "icon": "👕",
        "badge": "Reusable / Textile Recycling",
        "badge_color": "#8338ec",
        "bin": "Textile Bank / Charity Donation",
        "recyclable": True,
        "advice": (
            "If wearable, donate to thrift shops, charities, or clothing donation banks. "
            "If torn or stained, take to dedicated textile recycling centers."
        )
    },
    "glass": {
        "icon": "🍾",
        "badge": "100% Infinitely Recyclable",
        "badge_color": "#06d6a0",
        "bin": "Glass Recycling Bin",
        "recyclable": True,
        "advice": (
            "Glass can be recycled indefinitely with zero quality loss. "
            "Rinse out contents. Remove non-glass caps and place in designated glass recycling bin."
        )
    },
    "metal": {
        "icon": "🥫",
        "badge": "Highly Recyclable",
        "badge_color": "#118ab2",
        "bin": "Metal / Cans Recycling Bin",
        "recyclable": True,
        "advice": (
            "Rinse beverage cans, soup cans, and aluminum trays. "
            "Crush cans to save space. Metal can be recycled indefinitely."
        )
    },
    "paper": {
        "icon": "📰",
        "badge": "Recyclable",
        "badge_color": "#3a86ff",
        "bin": "Blue Paper Recycling Bin",
        "recyclable": True,
        "advice": (
            "Recycle office paper, newspapers, magazines, envelopes, and brochures. "
            "Keep paper clean and dry. Avoid recycling wax-coated paper or grease-soaked paper towels."
        )
    },
    "plastic": {
        "icon": "🧴",
        "badge": "Recyclable (Check Resin Code)",
        "badge_color": "#f4a261",
        "bin": "Yellow / Plastics Recycling Bin",
        "recyclable": True,
        "advice": (
            "Empty all liquids and rinse residue. Check local resin recycling codes. "
            "Screw caps back onto bottles and place in curbside recycling."
        )
    },
    "shoes": {
        "icon": "👟",
        "badge": "Reusable / Shoe Recycling",
        "badge_color": "#fb8500",
        "bin": "Footwear Donation / Drop-off",
        "recyclable": True,
        "advice": (
            "Tie shoes together by laces so pairs stay matched. Donate wearable shoes. "
            "Worn-out athletic shoes can be recycled through athletic shoe brand take-back programs."
        )
    },
    "trash": {
        "icon": "🗑️",
        "badge": "Non-Recyclable Residual",
        "badge_color": "#6c757d",
        "bin": "Black General Trash / Landfill",
        "recyclable": False,
        "advice": (
            "Non-recyclable composite materials, heavily soiled wrappers, ceramics, or sanitary waste. "
            "Dispose in regular household residual waste."
        )
    }
}

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "ResNet-18 (Transfer Learning - Recommended)": {
        "type": "resnet18",
        "filenames": ["best_waste_resnet18.pth", "models/best_waste_resnet18.pth"],
        "accuracy": "91.48%",
        "architecture": "ResNet-18 (Pretrained on ImageNet)",
        "description": "Deep 18-layer residual network with transfer learning."
    },
    "WasteCNN (Custom Baseline)": {
        "type": "cnn",
        "filenames": ["models/best_waste_cnn.pth", "best_waste_cnn.pth"],
        "accuracy": "42.95%",
        "architecture": "Custom 3-Layer CNN",
        "description": "Compact convolutional baseline trained from scratch."
    }
}

@st.cache_resource
def load_classifier(model_name: str):
    config = MODEL_CONFIGS[model_name]
    model_path = None
    for fname in config["filenames"]:
        p = BASE_DIR / fname
        if p.exists():
            model_path = p
            break

    if not model_path:
        raise FileNotFoundError(f"Checkpoint for '{model_name}' not found.")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    class_names = checkpoint.get("class_names")

    if config["type"] == "cnn":
        from src.cnn import WasteCNN
        model = WasteCNN(num_classes=len(class_names))
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    val_acc = checkpoint.get("validation_accuracy", None)
    return model, class_names, val_acc

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2b9348, #55a630, #80b918);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    selected_model_name = st.selectbox(
        "Select Model Architecture",
        list(MODEL_CONFIGS.keys()),
        index=0
    )

    model_info = MODEL_CONFIGS[selected_model_name]
    st.info(f"**Architecture:** {model_info['architecture']}\n\n"
            f"**Reported Accuracy:** `{model_info['accuracy']}`\n\n"
            f"{model_info['description']}")

    st.markdown("---")
    st.markdown("### 🖥️ Hardware Device")
    if DEVICE.type == "cuda":
        st.success(f"GPU Accelerated: `{torch.cuda.get_device_name(0)}`")
    else:
        st.info("Running on CPU (`Optimized for inference`)")

    st.markdown("---")
    st.markdown("### 🏷️ 10 Supported Categories")
    cols_a, cols_b = st.columns(2)
    categories_list = list(WASTE_DATA.keys())
    for idx, cat in enumerate(categories_list):
        target_col = cols_a if idx % 2 == 0 else cols_b
        target_col.caption(f"{WASTE_DATA[cat]['icon']} {cat.title()}")

    st.markdown("---")
    st.caption("EcoSort AI • Waste Classification System")

st.markdown("<div class='main-header'>♻️ EcoSort AI — Waste Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Upload an image of household or industrial waste to instantly classify its category and receive disposal & recycling instructions.</div>", unsafe_allow_html=True)

input_tab1, input_tab2 = st.tabs(["📁 Upload Image", "🖼️ Try Sample Images"])

input_image = None
image_source_label = ""

with input_tab1:
    uploaded_file = st.file_uploader(
        "Choose an image from your device",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
            image_source_label = uploaded_file.name
        except Exception as e:
            st.error(f"Error opening image: {e}")

with input_tab2:
    if SAMPLE_DIR.exists():
        sample_files = sorted(list(SAMPLE_DIR.glob("*_sample.jpg")))
        if sample_files:
            sample_names = [f.stem.replace("_sample", "").title() for f in sample_files]
            selected_sample_idx = st.selectbox(
                "Choose a sample waste item to test:",
                range(len(sample_files)),
                format_func=lambda i: f"{WASTE_DATA.get(sample_files[i].stem.replace('_sample', '').lower(), {}).get('icon', '📦')} {sample_names[i]}"
            )
            if st.button("Load Selected Sample", type="secondary"):
                selected_sample_path = sample_files[selected_sample_idx]
                input_image = Image.open(selected_sample_path).convert("RGB")
                image_source_label = f"Sample: {sample_names[selected_sample_idx]}"
                st.session_state["active_sample"] = selected_sample_path
        else:
            st.info("No sample images found in sample_images.")
    else:
        st.info("Sample directory not found.")

    if input_image is None and "active_sample" in st.session_state:
        if st.session_state["active_sample"].exists():
            input_image = Image.open(st.session_state["active_sample"]).convert("RGB")
            image_source_label = f"Sample: {st.session_state['active_sample'].stem.replace('_sample', '').title()}"

if input_image is not None:
    st.markdown("---")
    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        st.subheader("📸 Input Image")
        st.image(input_image, caption=image_source_label, use_container_width=True)

    with col_right:
        st.subheader("🔍 Classification Results")

        try:
            with st.spinner("Classifying waste category..."):
                model, class_names, loaded_val_acc = load_classifier(selected_model_name)
                input_tensor = IMAGE_TRANSFORM(input_image).unsqueeze(0).to(DEVICE)

                with torch.inference_mode():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]

                top_idx = probabilities.argmax().item()
                predicted_class = class_names[top_idx].lower()
                confidence = probabilities[top_idx].item() * 100

                meta = WASTE_DATA.get(predicted_class, {
                    "icon": "♻️",
                    "badge": "Classified",
                    "badge_color": "#457b9d",
                    "bin": "General Sorting",
                    "recyclable": True,
                    "advice": "Check your local municipal waste regulations."
                })

            st.markdown(
                f"""
                <div style="padding: 1.2rem; border-radius: 12px; background: {meta['badge_color']}18; border-left: 6px solid {meta['badge_color']}; margin-bottom: 1rem;">
                    <div style="font-size: 0.9rem; font-weight: 600; color: {meta['badge_color']}; text-transform: uppercase; letter-spacing: 0.5px;">
                        {meta['badge']}
                    </div>
                    <div style="font-size: 2rem; font-weight: 800; margin: 0.3rem 0;">
                        {meta['icon']} {predicted_class.title()}
                    </div>
                    <div style="font-size: 1.05rem; opacity: 0.9;">
                        Confidence: <b>{confidence:.1f}%</b> • Destination: <b>{meta['bin']}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.container():
                st.markdown(f"#### 💡 How to Dispose / Recycle")
                st.info(meta["advice"])

            st.markdown("#### 📊 Confidence Breakdown")
            scores = sorted(
                zip(class_names, probabilities.cpu().tolist()),
                key=lambda x: x[1],
                reverse=True
            )

            for cat, score in scores:
                cat_meta = WASTE_DATA.get(cat.lower(), {})
                cat_icon = cat_meta.get("icon", "•")
                percent = score * 100

                cols_bar = st.columns([2, 5, 2])
                cols_bar[0].write(f"{cat_icon} **{cat.title()}**")
                cols_bar[1].progress(score)
                cols_bar[2].caption(f"`{percent:5.1f}%`")

        except Exception as err:
            st.error(f"Inference error: {err}")
            st.exception(err)

else:
    st.markdown("---")
    st.info("👆 Please upload an image or select a sample image above to see real-time classification.")

    st.markdown("### 🌟 Project Highlights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🎯 91.5% Accuracy")
        st.write("ResNet-18 model fine-tuned on diverse waste items across 10 common categories.")
    with c2:
        st.markdown("#### 🌍 Eco-Friendly Guidance")
        st.write("Provides actionable disposal and recycling instructions for every classified material type.")
    with c3:
        st.markdown("#### ⚡ Real-Time Inference")
        st.write("Fast PyTorch inference on CPU or GPU with full probability distributions.")
