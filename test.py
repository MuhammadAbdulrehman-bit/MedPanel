import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import gradio as gr

# ==========================================
# 1. Model Architectures & Loading
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RETINAL_CLASSES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
SKIN_CLASSES = ['Melanoma', 'Melanocytic Nevus', 'Basal Cell Carcinoma', 
                'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion']

def load_retinal_model(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 5)
    )
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device).eval()

def load_skin_model(path):
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(1536, 512), nn.ReLU(),
        nn.Dropout(p=0.3), nn.Linear(512, 7)
    )
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device).eval()

# Load Models
try:
    retinal_model = load_retinal_model(r"E:\New folder\MediScan\backend\models\retinal_v2_best.pth")
    skin_model = load_skin_model(r"E:\New folder\MediScan\backend\models\skin_efficientnet_best (1).pth")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")

# ==========================================
# 2. Advanced Preprocessing (Gaussian Filter)
# ==========================================

def apply_gaussian_filter(pil_img):
    """Replicates the APTOS 2019 Gaussian filtering process."""
    img = np.array(pil_img.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    
    # Ben Graham's method: subtract local average color
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    
    # Convert back to PIL
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# Standard normalization for both models
standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. Prediction Logic
# ==========================================
def predict(image, scan_type):
    if image is None: return "Please upload an image."
    
    if scan_type == "Diabetic Retinopathy":
        # 1. Apply the Gaussian Filter first
        processed_img = apply_gaussian_filter(image)
        # 2. Convert to tensor
        img_tensor = standard_transform(processed_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = retinal_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            return {RETINAL_CLASSES[i]: float(probs[i]) for i in range(5)}
    else:
        # Skin model uses standard resize to 300 (B3 default)
        img_tensor = standard_transform(image.resize((300, 300))).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = skin_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            return {SKIN_CLASSES[i]: float(probs[i]) for i in range(7)}

# ==========================================
# 4. Gradio UI
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 MediScan Diagnostics")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil")
            scan_type = gr.Radio(["Skin Cancer (HAM10000)", "Diabetic Retinopathy"], 
                                 label="Model Selection", value="Skin Cancer (HAM10000)")
            btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            output = gr.Label(label="Diagnosis Result")

    btn.click(predict, inputs=[image_input, scan_type], outputs=output)

demo.launch(theme=gr.themes.Soft())