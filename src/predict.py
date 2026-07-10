import numpy as np
import tensorflow as tf
from src.utils import preprocess_image, assess_severity, get_treatment
from src.gradcam import make_gradcam_heatmap, overlay_gradcam, get_base_model
import json
import os


def load_class_names(path="model/class_names.json"):
    with open(path, "r") as f:
        return json.load(f)


def load_model(model_path="model/crop_disease_model.keras"):
    return tf.keras.models.load_model(model_path)


def predict(image, model, class_names, treatment_data, generate_gradcam=True):
    """
    Two-stage prediction pipeline:
    Stage 1 — Disease classification using EfficientNetB0
    Stage 2 — Severity assessment using visual features

    generate_gradcam: if True, attempts to add a 'gradcam_image' key
    (PIL Image) to the result. On any failure this is silently omitted
    rather than breaking the whole prediction -- Grad-CAM is a nice-to-have
    explanation layer, not a dependency of the core diagnosis.
    """
    # Preprocess — raw 0-255 pixels matching training
    processed = preprocess_image(image)

    # Stage 1 — Disease classification
    predictions = model.predict(processed, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    disease_class = class_names[predicted_idx]

    # Stage 2 — Severity assessment
    is_healthy = "healthy" in disease_class.lower()
    severity = "None" if is_healthy else assess_severity(image)

    # Treatment recommendation
    treatment_info = get_treatment(
        disease_class, severity, treatment_data
    )

    result = {
        "disease_class": disease_class,
        "common_name": treatment_info["common_name"],
        "confidence": confidence,
        "is_healthy": is_healthy,
        "severity": severity,
        "severity_info": treatment_info["severity_info"],
        "treatment": treatment_info["treatment"],
        "prevention": treatment_info["prevention"],
        "top_3": [
            {
                "class": class_names[i].replace(
                    "___", " — ").replace("_", " "),
                "confidence": float(predictions[0][i]) * 100
            }
            for i in np.argsort(predictions[0])[-3:][::-1]
        ]
    }

    if generate_gradcam:
        try:
            base_model = get_base_model(model)
            heatmap = make_gradcam_heatmap(
                processed, model, base_model, pred_index=predicted_idx
            )
            result["gradcam_image"] = overlay_gradcam(image, heatmap)
        except Exception as gradcam_error:
            # Don't let a Grad-CAM failure take down the whole prediction.
            # The app.py UI slot for gradcam_image is already conditional
            # on this key existing, so omitting it just hides that section.
            print(f"[gradcam] skipped: {gradcam_error}")

    return result