import os
from PIL import Image

from backend.models.xray_model import analyze_xray
from backend.models.skin_model import analyze_skin
from backend.models.retinal_model import analyze_retinal


def detect_image_type(image_path: str) -> str:
    """
    Auto-detects what type of medical image was uploaded
    based on basic image properties.
    Returns: 'xray', 'skin', or 'retinal'
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Sample pixels to detect image characteristics
    grayscale = image.convert("L")
    pixels = list(grayscale.getdata())
    avg_brightness = sum(pixels) / len(pixels)

    # X-rays are typically grayscale with high contrast
    # Check how "gray" the image is
    r, g, b = 0, 0, 0
    sample = list(image.getdata())[::100]  # sample every 100th pixel
    for pr, pg, pb in sample:
        r += pr
        g += pg
        b += pb
    n = len(sample)
    r, g, b = r / n, g / n, b / n

    # If R, G, B channels are very close = grayscale = likely X-ray
    channel_diff = max(abs(r - g), abs(g - b), abs(r - b))

    if channel_diff < 15 and avg_brightness < 130:
        return "xray"
    elif avg_brightness > 100 and channel_diff > 20:
        return "skin"
    else:
        return "retinal"


def run_vision_agent(image_path: str, hint: str = None) -> dict:
    """
    Agent 1 — Vision Triage Agent.
    Detects image type and runs the correct CV model.
    
    Args:
        image_path: path to uploaded image
        hint: optional user hint ('xray', 'skin', 'retinal')
    
    Returns:
        dict with scan_type, findings, risk_level, top_finding
    """
    # If user told us what type it is, trust them
    if hint and hint.lower() in ["xray", "skin", "retinal"]:
        scan_type = hint.lower()
    else:
        scan_type = detect_image_type(image_path)

    print(f"[Vision Agent] Detected scan type: {scan_type}")

    if scan_type == "xray":
        result = analyze_xray(image_path)
    elif scan_type == "skin":
        result = analyze_skin(image_path)
    elif scan_type == "retinal":
        result = analyze_retinal(image_path)
    else:
        result = analyze_xray(image_path)  # default fallback

    print(f"[Vision Agent] Top finding: {result['top_finding']} | Risk: {result['risk_level']}")
    return result