import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from torchvision import transforms
from PIL import Image
import os

SKIN_CLASSES = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion"
]

SKIN_TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "skin_efficientnet_best.pth"
)


def build_skin_model():
    model = efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, len(SKIN_CLASSES))
    )

    if os.path.exists(WEIGHTS_PATH):
        print("[Skin Model] Loading trained EfficientNetB3 weights...")
        ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print("[Skin Model] Weights loaded successfully")
    else:
        print(f"[Skin Model] WARNING: weights not found at {WEIGHTS_PATH}")

    return model.to(device)


def analyze_skin(image_path: str) -> dict:
    model = build_skin_model()
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = SKIN_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    findings = []
    for i, prob in enumerate(probs):
        findings.append({
            "condition": SKIN_CLASSES[i],
            "confidence": round(float(prob), 3)
        })

    findings.sort(key=lambda x: x["confidence"], reverse=True)
    top = findings[0]

    high_risk = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratosis"]
    medium_risk = ["Benign Keratosis", "Vascular Lesion"]

    if top["condition"] in high_risk and top["confidence"] > 0.5:
        risk = "High"
    elif top["condition"] in medium_risk:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "scan_type": "skin_lesion",
        "findings": findings[:4],
        "risk_level": risk,
        "top_finding": top["condition"]
    }