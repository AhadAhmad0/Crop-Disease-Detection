# 🌿 AgroScan AI — Crop Disease Detection

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3-D00000?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?logo=docker&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-yellow)

An end-to-end deep learning application that detects crop leaf diseases from images, assesses severity, explains its predictions visually, and recommends treatment — deployed as a live web app.

**[Live Demo →](https://huggingface.co/spaces/AhadAhmad0/crop-disease-detection)**

![AgroScan AI interface](screenshots/app-overview.png)

---

## Overview

AgroScan AI classifies leaf images into 14 disease categories across 5 crops (Apple, Corn, Grape, Potato, Tomato) using a fine-tuned EfficientNetB0 model, then layers on three things that turn a raw classifier into something actually usable in the field:

- **Severity assessment** — grades disease progression (Early / Moderate / Advanced) rather than just naming the disease
- **Grad-CAM explainability** — visually highlights the exact leaf regions the model used to make its prediction, so the output isn't a black-box label
- **Treatment recommendations** — maps each disease + severity combination to actionable, evidence-based treatment and prevention steps

The model reaches **99.55% validation accuracy** (macro F1: 99.55%) on 6,667 held-out images.

---

## Architecture

### Model

- **Backbone:** EfficientNetB0, ImageNet-pretrained, `include_top=False`
- **Training strategy:** two-phase transfer learning
  - **Phase 1** — backbone frozen, only the classification head trains (fast convergence, stabilizes the new head before touching pretrained weights)
  - **Phase 2** — last 30 layers of the backbone unfrozen and fine-tuned at a low learning rate (`1e-5`)
- **Classification head:** `GlobalAveragePooling2D → BatchNormalization → Dense(512, relu) → Dropout(0.4) → Dense(256, relu) → Dropout(0.3) → Dense(num_classes, softmax)`
- **Input:** 224×224×3, **raw 0–255 pixel values** (not normalized — EfficientNet's preprocessing expects this range; feeding normalized inputs silently collapses accuracy to ~7%, a specific gotcha this project's training pipeline accounts for)
- **Dataset:** 14 selected classes from the PlantVillage / New Plant Diseases Dataset

### Explainability — Grad-CAM

Grad-CAM highlights which pixels most influenced a given prediction by tracing gradients back to the last convolutional activation layer (`top_activation` in EfficientNetB0).

This isn't a drop-in application of the standard Keras Grad-CAM recipe — EfficientNetB0 here is a **nested sub-model** inside the outer classifier, and Keras 3 treats a nested-model call as an opaque graph node. That breaks the usual approach of building one merged `Model` with the conv layer and the final output as joint outputs (`Output ... is not connected to inputs`). The implementation instead:

1. Builds a small model scoped to the base model's own graph, mapping its input to the target conv layer's output
2. Manually re-applies the classification head's layers (GAP, BatchNorm, Dense, Dropout, ...) on top of that conv output, inside a single `GradientTape`
3. This keeps everything in one connected differentiable chain, so gradients flow correctly from the predicted class back to the conv activations

![Grad-CAM example](screenshots/gradcam-example.png)

### Application

- **Frontend:** Streamlit, with a custom design system (CSS variables, card components, severity badges) rather than default Streamlit styling
- **Inference pipeline:** `src/predict.py` orchestrates preprocessing → model prediction → severity assessment → Grad-CAM generation → treatment lookup, returning a single result object the UI renders from
- **Deployment:** containerized with Docker, deployed on Hugging Face Spaces running `tensorflow-cpu` (no GPU needed at inference time)

---

## Project Structure

```
crop-disease-detection/
├── app.py                      # Streamlit UI — layout, styling, result rendering
├── model.py                    # Model architecture + two-phase training logic
├── src/
│   ├── predict.py              # Inference pipeline: preprocess → predict → Grad-CAM → treatment
│   ├── gradcam.py              # Grad-CAM heatmap generation + overlay
│   └── utils.py                # Image preprocessing, severity scoring, treatment data loading
├── model/
│   ├── crop_disease_model.keras   # Trained model weights (tracked via Git LFS)
│   └── class_names.json           # Class index → disease name mapping
├── screenshots/                # README images
├── treatment_data.json         # Disease/severity → treatment & prevention mapping
├── Dockerfile                  # Container definition for Hugging Face Spaces deployment
├── requirements.txt            # Local development dependencies
└── requirements_hf.txt         # Deployment dependencies (tensorflow-cpu build)
```

---

## Key Engineering Decisions & Fixes

A few non-obvious issues surfaced during development that are worth documenting for anyone extending this project:

- **Input normalization bug:** EfficientNetB0 in this Keras/TF version expects raw `[0, 255]` pixel input, not normalized `[0, 1]` — feeding normalized images trained a model that silently converged to ~7% accuracy with no errors thrown. Root-caused by comparing preprocessing against the exact function EfficientNet expects internally.
- **Lambda layer serialization:** custom `Lambda` layers in the model definition failed to serialize/deserialize correctly under Keras 3 and were removed in favor of standard layer equivalents.
- **Grad-CAM under Keras 3:** the legacy `layer.get_output_at()` API used in most Grad-CAM tutorials was removed in Keras 3, and nested-model architectures additionally break the standard merged-`Model` Grad-CAM pattern with a disconnected-graph error. See the Explainability section above for the fix.
- **Large model file:** the 38.5MB `.keras` weights file is tracked with Git LFS rather than committed directly, keeping the repository lightweight to clone.

---

## Tech Stack

`TensorFlow / Keras 3` · `EfficientNetB0` · `Streamlit` · `OpenCV` · `Docker` · `Hugging Face Spaces` · `Git LFS`

---

## Author

**Ahad Ahmad**
B.Tech CS (Data Science & AI), Shri Ramswaroop Memorial University
[GitHub](https://github.com/AhadAhmad0) · ahadahmad0701@gmail.com
