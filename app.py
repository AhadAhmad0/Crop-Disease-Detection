import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.utils import preprocess_image, assess_severity, get_treatment, load_treatment_data
from src.predict import predict

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4caf50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #aaa;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #1e3a2f;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #e0e0e0;
    }
    .result-box h3 { color: #81c784; margin-bottom: 0.5rem; }
    .result-box p { color: #e0e0e0; margin: 0.2rem 0; }
    .healthy-box {
        background: #1b3a1f;
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        border-radius: 4px;
        color: #e0e0e0;
    }
    .healthy-box h3 { color: #81c784; margin-bottom: 0.5rem; }
    .healthy-box p { color: #e0e0e0; margin: 0.2rem 0; }
    .severity-Early {
        background: #3a3000;
        border-left: 5px solid #f9a825;
        padding: 0.8rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #fff9c4;
    }
    .severity-Moderate {
        background: #3a1f00;
        border-left: 5px solid #ef6c00;
        padding: 0.8rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #ffe0b2;
    }
    .severity-Advanced {
        background: #3a0000;
        border-left: 5px solid #c62828;
        padding: 0.8rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #ffcdd2;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    model_path = os.path.join(BASE_DIR, "model", "crop_disease_model.keras")
    class_path = os.path.join(BASE_DIR, "model", "class_names.json")
    treatment_path = os.path.join(BASE_DIR, "treatment_data.json")

    model = tf.keras.models.load_model(model_path)

    with open(class_path, "r") as f:
        class_names = json.load(f)

    treatment_data = load_treatment_data(treatment_path)
    return model, class_names, treatment_data

# Header
st.markdown(
    '<div class="main-header">🌿 Crop Disease Detection</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Upload a leaf image to detect disease, '
    'assess severity and get treatment recommendations</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown("### About This App")
    st.markdown("""
    **Two-stage deep learning pipeline:**

    **Stage 1 — Disease Classification**
    EfficientNetB0 trained on 26,000+ leaf images
    across 14 crop disease classes

    **Stage 2 — Severity Assessment**
    Analyses lesion area and colour distribution
    to grade disease progression

    **Treatment Recommendation**
    Evidence-based treatment mapped to each
    disease and severity level
    """)
    st.markdown("---")
    st.markdown("**Supported Crops:**")
    st.markdown("🍎 Apple | 🌽 Corn | 🍇 Grape | 🥔 Potato | 🍅 Tomato")
    st.markdown("---")
    st.markdown("**Model Performance:**")
    st.markdown("✅ Accuracy: 99.55%")
    st.markdown("✅ Macro F1: 99.55%")
    st.markdown("✅ Classes: 14")
    st.markdown("✅ Validation images: 6,667")

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file:
        st.markdown("### Analysis Results")
        with st.spinner("Analyzing leaf image..."):
            try:
                model, class_names, treatment_data = load_model_and_data()
                result = predict(image, model, class_names, treatment_data)

                if result["is_healthy"]:
                    st.markdown(
                        f'<div class="healthy-box">'
                        f'<h3>✅ Plant Appears Healthy</h3>'
                        f'<p><strong>Crop:</strong> '
                        f'{result["common_name"]}</p>'
                        f'<p><strong>Confidence:</strong> '
                        f'{result["confidence"]:.1f}%</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.success(f"Prevention: {result['prevention']}")
                else:
                    st.markdown(
                        f'<div class="result-box">'
                        f'<h3>🔍 Disease Detected</h3>'
                        f'<p><strong>Disease:</strong> '
                        f'{result["common_name"]}</p>'
                        f'<p><strong>Confidence:</strong> '
                        f'{result["confidence"]:.1f}%</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    severity_emoji = {
                        "Early": "🟡",
                        "Moderate": "🟠",
                        "Advanced": "🔴"
                    }.get(result["severity"], "⚪")

                    st.markdown(
                        f'<div class="severity-{result["severity"]}">'
                        f'<strong>{severity_emoji} Severity: '
                        f'{result["severity"]}</strong><br>'
                        f'{result["severity_info"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown("#### 💊 Recommended Treatment")
                    for i, step in enumerate(result["treatment"], 1):
                        st.markdown(f"{i}. {step}")

                    st.markdown("#### 🛡️ Prevention")
                    st.info(result["prevention"])

                st.markdown("#### Prediction Confidence")
                st.progress(result["confidence"] / 100)
                st.caption(f"{result['confidence']:.1f}% confident")

                with st.expander("See top 3 predictions"):
                    for item in result["top_3"]:
                        st.markdown(
                            f"- **{item['class']}**: "
                            f"{item['confidence']:.1f}%"
                        )

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please try uploading a different image.")
    else:
        st.info("👈 Upload a leaf image to get started")
        st.markdown("""
        **After uploading you will see:**
        - ✅ Disease name and confidence score
        - 🟡 Severity — Early / Moderate / Advanced
        - 💊 Step-by-step treatment recommendations
        - 🛡️ Prevention advice
        - 📊 Top 3 predictions
        """)

st.markdown("---")
st.markdown(
    "Built by Ahad Ahmad | "
    "[GitHub](https://github.com/AhadAhmad0/crop-disease-detection) | "
    "Model: EfficientNetB0 | Dataset: PlantVillage (14 classes)"
)