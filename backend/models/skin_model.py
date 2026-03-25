import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_skin_model():
    model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    model.classifier[1] = nn.Linear(model.last_channel, len(SKIN_CLASSES))
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
        if prob > 0.1:
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
        "findings": findings[:3],
        "risk_level": risk,
        "top_finding": top["condition"]
    }