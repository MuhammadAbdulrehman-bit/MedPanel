import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

RETINAL_CLASSES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

RETINAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_retinal_model():
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(RETINAL_CLASSES))
    return model.to(device)


def analyze_retinal(image_path: str) -> dict:
    model = build_retinal_model()
    model.eval()

    image = Image.open(image_path).convert("RGB")
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