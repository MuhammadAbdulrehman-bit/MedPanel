import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

RETINAL_CLASSES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "retinal_v2_best.pth"
)


def apply_gaussian_filter(pil_img):
    """Replicates training preprocessing — Ben Graham's method."""
    img = np.array(pil_img.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


RETINAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def build_retinal_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(RETINAL_CLASSES))
    )

    if os.path.exists(WEIGHTS_PATH):
        print("[Retinal Model] Loading trained ResNet50 weights...")
        ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print("[Retinal Model] Weights loaded successfully")
    else:
        print(f"[Retinal Model] WARNING: weights not found at {WEIGHTS_PATH}")

    return model.to(device)


def analyze_retinal(image_path: str) -> dict:
    model = build_retinal_model()
    model.eval()

    # Apply same Gaussian preprocessing used during training
    image = Image.open(image_path).convert("RGB")
    image = apply_gaussian_filter(image)
    tensor = RETINAL_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    findings = []
    for i, prob in enumerate(probs):
        findings.append({
            "condition": RETINAL_CLASSES[i],
            "confidence": round(float(prob), 3)
        })

    findings.sort(key=lambda x: x["confidence"], reverse=True)
    top = findings[0]

    risk_map = {
        "No DR": "Low",
        "Mild DR": "Low",
        "Moderate DR": "Medium",
        "Severe DR": "High",
        "Proliferative DR": "High"
    }

    return {
        "scan_type": "retinal",
        "findings": findings,
        "risk_level": risk_map[top["condition"]],
        "top_finding": top["condition"]
    }