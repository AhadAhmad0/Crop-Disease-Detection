import numpy as np
from PIL import Image
import json
import os

def load_treatment_data(json_path="treatment_data.json"):
    with open(json_path, "r") as f:
        return json.load(f)

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for EfficientNetB0 prediction.
    Returns raw 0-255 pixel values — matches training pipeline.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    image = image.resize(target_size)
    # Keep 0-255 range — model trained on raw pixels
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def assess_severity(image):
    """
    Assess disease severity using visual features.
    Returns: Early / Moderate / Advanced
    """
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image.copy()

    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    # Dark pixel ratio — diseased areas are darker
    gray = np.mean(img_array, axis=2)
    dark_pixels = np.sum(gray < 80)
    total_pixels = gray.size
    darkness_ratio = dark_pixels / total_pixels

    # Brown/yellow lesion ratio
    brown_mask = (r > 100) & (g < 100) & (b < 80)
    yellow_mask = (r > 150) & (g > 120) & (b < 80)
    lesion_ratio = (np.sum(brown_mask) + np.sum(yellow_mask)) / total_pixels

    severity_score = (darkness_ratio * 0.4) + (lesion_ratio * 0.6)

    if severity_score < 0.08:
        return "Early"
    elif severity_score < 0.20:
        return "Moderate"
    else:
        return "Advanced"

def get_treatment(disease_class, severity, treatment_data):
    if disease_class in treatment_data:
        data = treatment_data[disease_class]
    else:
        return {
            "common_name": disease_class.replace("___", " — ").replace("_", " "),
            "severity_info": "Consult an agricultural expert for assessment.",
            "treatment": ["Consult a local agricultural extension officer"],
            "prevention": "Monitor crops regularly and maintain field hygiene"
        }

    severity_info = data["severity_description"].get(
        severity, "Assess manually."
    )

    return {
        "common_name": data["common_name"],
        "severity_info": severity_info,
        "treatment": data["treatment"],
        "prevention": data["prevention"]
    }