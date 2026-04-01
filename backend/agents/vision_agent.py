import os
from PIL import Image

from backend.models.xray_model import analyze_xray
from backend.models.skin_model import analyze_skin
from backend.models.retinal_model import analyze_retinal

# Valid hint values from frontend
VALID_HINTS = ["xray", "skin", "skin_lesion", "retinal"]


def detect_image_type(image_path: str) -> str:
    """
    Auto-detects medical image type based on pixel characteristics.
    Returns: 'xray', 'skin_lesion', or 'retinal'
    NOTE: This is a heuristic fallback. User-provided hint is always preferred.
    """
    image = Image.open(image_path).convert("RGB")

    grayscale = image.convert("L")
    pixels = list(grayscale.getdata())
    avg_brightness = sum(pixels) / len(pixels)

    sample = list(image.getdata())[::100]
    r = sum(p[0] for p in sample) / len(sample)
    g = sum(p[1] for p in sample) / len(sample)
    b = sum(p[2] for p in sample) / len(sample)

    channel_diff = max(abs(r - g), abs(g - b), abs(r - b))

    if channel_diff < 15 and avg_brightness < 130:
        detected = "xray"
    elif avg_brightness > 100 and channel_diff > 20:
        detected = "skin_lesion"
    else:
        detected = "retinal"

    print(f"[Vision Agent] Auto-detected scan type: {detected} "
          f"(brightness={avg_brightness:.1f}, channel_diff={channel_diff:.1f})")
    return detected


def normalize_hint(hint: str) -> str:
    """
    Normalize user hint to internal scan type format.
    Frontend sends: 'xray', 'skin', 'retinal'
    Internal format: 'xray', 'skin_lesion', 'retinal'
    """
    hint = hint.lower().strip()
    mapping = {
        "xray": "xray",
        "chest_xray": "xray",
        "skin": "skin_lesion",
        "skin_lesion": "skin_lesion",
        "retinal": "retinal",
        "eye": "retinal",
        "dr": "retinal",
    }
    return mapping.get(hint, None)


def run_vision_agent(image_path: str, hint: str = None) -> dict:
    """
    Agent 1 — Vision Triage Agent.
    Routes image to correct CV model based on scan type.

    Args:
        image_path: path to uploaded image
        hint: optional user hint from frontend

    Returns:
        dict with scan_type, findings, risk_level, top_finding
    """
    # Normalize and validate hint
    if hint:
        scan_type = normalize_hint(hint)
        if scan_type:
            print(f"[Vision Agent] Using user hint: {hint} → {scan_type}")
        else:
            print(f"[Vision Agent] Unknown hint '{hint}', falling back to auto-detect")
            scan_type = detect_image_type(image_path)
    else:
        print(f"[Vision Agent] No hint provided, using auto-detect")
        scan_type = detect_image_type(image_path)

    # Route to correct CV model
    if scan_type == "xray":
        result = analyze_xray(image_path)
    elif scan_type == "skin_lesion":
        result = analyze_skin(image_path)
    elif scan_type == "retinal":
        result = analyze_retinal(image_path)
    else:
        print(f"[Vision Agent] Unknown scan type '{scan_type}', defaulting to xray")
        result = analyze_xray(image_path)

    # Ensure consistent scan_type in result
    result["scan_type"] = scan_type

    print(f"[Vision Agent] Top finding: {result['top_finding']} | "
          f"Risk: {result['risk_level']} | Scan: {scan_type}")
    return result