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
    page_title="AgroScan AI - Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Stitch design system -> CSS (colors/typography pulled from DESIGN.md)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(160deg, #e5f3ea 0%, #f8f9ff 45%, #ffffff 100%);
    }

    /* Header */
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        color: #022719;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #414844;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Generic white diagnostic card */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0px 4px 20px rgba(26, 61, 46, 0.05);
        margin-bottom: 1rem;
    }
    .card h3 {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #414844;
        margin-bottom: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .card .diagnosis-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: #022719;
    }

    /* Severity badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .badge-Early { background: #fea619; color: #684000; }
    .badge-Moderate { background: #ef6c00; color: #ffffff; }
    .badge-Advanced { background: #ba1a1a; color: #ffffff; }
    .badge-healthy { background: #00b853; color: #ffffff; }

    /* Confidence bar */
    .conf-track {
        width: 100%;
        background: #c1c8c2;
        height: 6px;
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .conf-fill {
        height: 100%;
        border-radius: 9999px;
        background: #00b853;
    }
    .conf-label {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #414844;
    }

    /* Symptom quote */
    .symptom-quote {
        border-left: 4px solid #fea619;
        padding-left: 0.75rem;
        font-style: italic;
        color: #414844;
        margin-top: 0.75rem;
    }

    /* Stat tiles */
    .stat-tile {
        background: #1a3d2e;
        color: #ffffff;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
    }
    .stat-tile p.label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        opacity: 0.8;
        margin-bottom: 0.2rem;
    }
    .stat-tile p.value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }

    /* Crop chips */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(2, 39, 25, 0.08);
        color: #022719;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin: 0.15rem;
    }

    /* Reserve fixed height for the results panel so nothing reflows/shakes
       between the empty state, the analyzing state, and the populated state */
    .results-shell {
        min-height: 620px;
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


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🌿 AgroScan AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a leaf image to detect disease, '
    'assess severity, and get treatment advice.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
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

    st.markdown("**Model Performance**")
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.markdown('<div class="stat-tile"><p class="label">Accuracy</p><p class="value">99.55%</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-tile"><p class="label">Classes</p><p class="value">14</p></div>', unsafe_allow_html=True)
    with stat_col2:
        st.markdown('<div class="stat-tile"><p class="label">Macro F1</p><p class="value">99.55%</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-tile"><p class="label">Validation</p><p class="value">6,667</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Supported Crops**")
    st.markdown(
        '<span class="chip">🍎 Apple</span>'
        '<span class="chip">🌽 Corn</span>'
        '<span class="chip">🍇 Grape</span>'
        '<span class="chip">🥔 Potato</span>'
        '<span class="chip">🍅 Tomato</span>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", use_container_width=True)

with col2:
    st.markdown("#### Analysis Results")

    # Fixed-height shell: prevents the card layout from jumping between
    # empty / loading / populated states (this was the cause of the UI
    # "shaking" you saw in the recording).
    results_shell = st.container()

    with results_shell:
        st.markdown('<div class="results-shell">', unsafe_allow_html=True)

        if uploaded_file:
            with st.spinner("Analyzing leaf image..."):
                try:
                    model, class_names, treatment_data = load_model_and_data()
                    result = predict(image, model, class_names, treatment_data)

                    is_healthy = result["is_healthy"]
                    confidence = result["confidence"]
                    name = result["common_name"]

                    # --- Diagnosis card -------------------------------------------------
                    if is_healthy:
                        badge_html = '<span class="badge badge-healthy">Healthy</span>'
                    else:
                        severity = result["severity"]
                        badge_html = f'<span class="badge badge-{severity}">{severity} Severity</span>'

                    st.markdown(f"""
                    <div class="card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <div>
                                <h3>Diagnosis</h3>
                                <p class="diagnosis-name">{'✅ Healthy — ' + name if is_healthy else name}</p>
                            </div>
                            {badge_html}
                        </div>
                        <div class="conf-label">
                            <span>Model Confidence</span>
                            <span>{confidence:.1f}%</span>
                        </div>
                        <div class="conf-track">
                            <div class="conf-fill" style="width:{confidence:.1f}%;"></div>
                        </div>
                        {f'<p class="symptom-quote">{result["severity_info"]}</p>' if not is_healthy else ''}
                    </div>
                    """, unsafe_allow_html=True)

                    # --- Top 3 predictions -----------------------------------------------
                    with st.expander("Top 3 Predictions", expanded=False):
                        for item in result["top_3"]:
                            st.markdown(
                                f"**{item['class']}** — {item['confidence']:.2f}%"
                            )

                    # --- Treatment / Prevention ------------------------------------------
                    if is_healthy:
                        st.markdown(f"""
                        <div class="card">
                            <h3>🛡️ Prevention</h3>
                            <p>{result['prevention']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        t_col1, t_col2 = st.columns(2)
                        with t_col1:
                            treatment_items = "".join(
                                f"<li>✅ {step}</li>" for step in result["treatment"]
                            )
                            st.markdown(f"""
                            <div class="card">
                                <h3>💊 Treatment</h3>
                                <ul style="padding-left:1rem; margin:0;">{treatment_items}</ul>
                            </div>
                            """, unsafe_allow_html=True)
                        with t_col2:
                            st.markdown(f"""
                            <div class="card">
                                <h3>🛡️ Prevention</h3>
                                <p>{result['prevention']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # --- Grad-CAM slot (conditional — only renders once your pipeline
                    # actually produces a gradcam image; otherwise omitted rather than
                    # shown as a fake/placeholder feature) -----------------------------
                    if "gradcam_image" in result:
                        st.markdown("#### Model Attention (Grad-CAM)")
                        st.image(
                            result["gradcam_image"],
                            caption="Highlighted regions influenced the prediction",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please try uploading a different image.")
        else:
            st.markdown("""
            <div class="card">
                <p>👈 Upload a leaf image to get started</p>
                <p style="margin-top:0.75rem; color:#414844;">
                    <strong>After uploading you will see:</strong><br>
                    ✅ Disease name and confidence score<br>
                    🟡 Severity — Early / Moderate / Advanced<br>
                    💊 Step-by-step treatment recommendations<br>
                    🛡️ Prevention advice<br>
                    📊 Top 3 predictions
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "**AgroScan AI** &nbsp;|&nbsp; Built by Ahad Ahmad &nbsp;|&nbsp; "
    "[GitHub](https://github.com/AhadAhmad0/crop-disease-detection) &nbsp;|&nbsp; "
    "Model: EfficientNetB0 &nbsp;|&nbsp; Dataset: PlantVillage (14 classes)"
)
