import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

XRAY_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

device = torch.device("cpu")

XRAY_TRANSFORM = transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])


def build_xray_model():
    print("[XRay Model] Loading TorchXRayVision pretrained model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    print("[XRay Model] Model loaded successfully")
    return model


def analyze_xray(image_path: str) -> dict:
    model = build_xray_model()

    # Load and preprocess image
    image = Image.open(image_path).convert("L")  # grayscale
    image_np = np.array(image)

    # Normalize to [-1024, 1024] range expected by TorchXRayVision
    image_np = xrv.datasets.normalize(image_np, maxval=255, reshape=True)

    # Apply transforms
    image_np = XRAY_TRANSFORM(image_np)
    tensor = torch.from_numpy(image_np).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)

    # Map TorchXRayVision pathology outputs to our classes
    pathologies = model.pathologies
    probs = outputs[0].numpy()

    findings = []
    for i, pathology in enumerate(pathologies):
        prob = float(probs[i])
        if prob > 0.15:
            findings.append({
                "condition": pathology,
                "confidence": round(prob, 3)
            })

    findings.sort(key=lambda x: x["confidence"], reverse=True)

    if not findings:
        findings = [{"condition": "No significant findings", "confidence": 1.0}]

    if any(f["confidence"] > 0.5 for f in findings):
        risk = "High"
    elif any(f["confidence"] > 0.3 for f in findings):
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "scan_type": "chest_xray",
        "findings": findings,
        "risk_level": risk,
        "top_finding": findings[0]["condition"]
    }