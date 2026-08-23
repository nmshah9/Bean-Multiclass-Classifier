"""
app.py
======
STEP 10 of the pipeline: Build a Simple Classifier App (Streamlit).

Why this app can safely assume the model already exists:
    All the heavy lifting (EDA, cleaning, training, tuning, evaluation) was
    done once in train_pipeline.py, which saved everything this app needs
    into the models/ folder. This app's ONLY job is: take a bean's physical
    measurements, apply the exact same scaling used at training time, and
    ask the exact same trained model for a prediction. Keeping the app this
    thin is deliberate -- it starts fast, has almost nothing that can break,
    and never risks re-training a different model than the one that was
    actually evaluated.

Run with:
    streamlit run app.py
(after running `python train_pipeline.py` once, if models/ is empty)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import (
    SCALER_PATH, LABEL_ENCODER_PATH, BEST_MODEL_PATH, FEATURE_NAMES_PATH,
    METADATA_PATH, RAW_DATA_PATH,
)

# ---------------------------------------------------------------------------
# Page config + Aeries brand styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dry Bean Classifier | Aeries Technology",
    page_icon="🫘",
    layout="wide",
)

BANNER_PATH = "banner.png"  # adjust to your actual path

banner = Image.open(BANNER_PATH)

# Resize if needed
banner = banner.resize((1800, 450))
# Display centered using Markdown wrapper
col1, col2, col3 = st.columns([0.5,5,1])  # middle column wider
with col2:
    st.image(banner, width=1800)
AERIES_NAVY = "#0B2545"
AERIES_TEAL = "#0F8B8D"
AERIES_CYAN = "#13C4C4"

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: #F7F9FA; }}
        .aeries-header {{
            background: linear-gradient(90deg, {AERIES_NAVY} 0%, {AERIES_TEAL} 100%);
            padding: 1.4rem 1.8rem;
            border-radius: 10px;
            margin-bottom: 1.4rem;
        }}
        .aeries-header h1 {{ color: white; margin: 0; font-size: 1.6rem; }}
        .aeries-header p {{ color: #E4F5F5; margin: 0.25rem 0 0 0; font-size: 0.95rem; }}
        div.stButton > button:first-child {{
            background-color: {AERIES_TEAL};
            color: white;
            border-radius: 6px;
            border: none;
        }}
        div.stButton > button:first-child:hover {{
            background-color: {AERIES_CYAN};
            color: {AERIES_NAVY};
        }}
        .metric-card {{
            background: white;
            border-left: 5px solid {AERIES_TEAL};
            padding: 0.9rem 1.1rem;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
    </style>
    <div class="aeries-header">
        <h1>🫘 Dry Bean Type Classifier</h1>
        <p>Automated multiclass classification of dry bean types from physical measurements</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Load the trained model + preprocessing artifacts once per app session.

    Why @st.cache_resource: joblib.load on a model file is relatively slow
    and the model/scaler/encoder never change while the app is running --
    caching means every user interaction (button click, slider drag) reuses
    the already-loaded objects instead of re-reading files from disk on
    every single rerun, which is how Streamlit apps stay responsive.
    """
    missing = [p for p in [SCALER_PATH, LABEL_ENCODER_PATH, BEST_MODEL_PATH, FEATURE_NAMES_PATH]
               if not os.path.exists(p)]
    if missing:
        return None

    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)

    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    return {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "model": model,
        "feature_names": feature_names,
        "metadata": metadata,
    }


artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "Model artifacts were not found in the `models/` folder.\n\n"
        "This app is designed to run **after** training has been done once. "
        "From the project root, run:\n\n"
        "```bash\npython train_pipeline.py\n```\n\n"
        "That script trains and compares every classifier, selects the best "
        "one, and saves everything this app needs "
        "(`best_model.joblib`, `scaler.joblib`, `label_encoder.joblib`, "
        "`feature_names.json`) into the `models/` folder. Once that "
        "finishes, refresh this page."
    )
    st.stop()

model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]
feature_names = artifacts["feature_names"]
metadata = artifacts["metadata"]

# Friendly units/help text per feature, shown next to each input so a
# non-technical user still understands what number they're entering.
FEATURE_HELP = {
    "Area": "Number of pixels inside the bean's boundary",
    "Perimeter": "Bean circumference (border length)",
    "MajorAxisLength": "Length of the longest line across the bean",
    "MinorAxisLength": "Longest line perpendicular to the major axis",
    "AspectRation": "Relationship between major and minor axis length",
    "Eccentricity": "Eccentricity of the ellipse with the same moments as the bean",
    "ConvexArea": "Pixels in the smallest convex polygon containing the bean",
    "EquivDiameter": "Diameter of a circle with the same area as the bean",
    "Extent": "Ratio of bean pixels to its bounding-box pixels",
    "Solidity": "Ratio of bean pixels to its convex-hull pixels",
    "roundness": "4*pi*Area / Perimeter^2",
    "Compactness": "EquivDiameter / MajorAxisLength",
    "ShapeFactor1": "Shape descriptor 1",
    "ShapeFactor2": "Shape descriptor 2",
    "ShapeFactor3": "Shape descriptor 3",
    "ShapeFactor4": "Shape descriptor 4",
}

# Reasonable default values + slider ranges derived from the training data's
# typical range, so a first-time user gets a sensible starting point instead
# of a blank/zero form that would produce a meaningless prediction.
DEFAULTS = {
    "Area": (28395.0, 20000, 254616),
    "Perimeter": (610.29, 500.0, 2100.0),
    "MajorAxisLength": (208.18, 150.0, 750.0),
    "MinorAxisLength": (173.89, 100.0, 460.0),
    "AspectRation": (1.20, 1.0, 2.5),
    "Eccentricity": (0.55, 0.0, 0.95),
    "ConvexArea": (28715.0, 20000.0, 263261.0),
    "EquivDiameter": (190.14, 150.0, 570.0),
    "Extent": (0.76, 0.5, 0.87),
    "Solidity": (0.99, 0.9, 1.0),
    "roundness": (0.96, 0.4, 1.0),
    "Compactness": (0.91, 0.6, 1.0),
    "ShapeFactor1": (0.0073, 0.002, 0.011),
    "ShapeFactor2": (0.0031, 0.0005, 0.004),
    "ShapeFactor3": (0.834, 0.4, 0.98),
    "ShapeFactor4": (0.9987, 0.9, 1.0),
}

tab_manual, tab_batch, tab_about = st.tabs(
    ["🔢 Manual Prediction", "📄 Batch Prediction (CSV)", "ℹ️ About This Model"]
)

# ---------------------------------------------------------------------------
# TAB 1: Manual single-bean prediction
# ---------------------------------------------------------------------------
with tab_manual:
    st.subheader("Enter a bean's physical measurements")
    st.caption(
        "Values default to a typical SEKER bean. Adjust the sliders to match "
        "your camera-based measurements, then click Predict."
    )

    input_values = {}
    cols = st.columns(4)
    for i, feat in enumerate(feature_names):
        default, lo, hi = DEFAULTS.get(feat, (0.0, 0.0, 1.0))
        with cols[i % 4]:
            input_values[feat] = st.number_input(
                feat,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                help=FEATURE_HELP.get(feat, ""),
                key=f"manual_{feat}",
            )

    predict_clicked = st.button("🔍 Predict Bean Type", use_container_width=True)

    if predict_clicked:
        # Build the input row in the EXACT column order the scaler was
        # fitted on (feature_names.json) -- this is what prevents the
        # classic "silent wrong predictions from misaligned columns" bug.
        input_df = pd.DataFrame([[input_values[f] for f in feature_names]], columns=feature_names)
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]

        st.markdown("### Prediction Result")
        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="color:{AERIES_NAVY}; font-size:0.9rem;">Predicted Bean Class</div>
                    <div style="color:{AERIES_TEAL}; font-size:1.8rem; font-weight:700;">{predicted_class}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_col2:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(scaled_input)[0]
                proba_df = pd.DataFrame({
                    "Bean Class": label_encoder.classes_,
                    "Confidence": proba,
                }).sort_values("Confidence", ascending=False)
                st.bar_chart(proba_df.set_index("Bean Class"))
                top_conf = proba_df.iloc[0]["Confidence"]
                st.caption(f"Model confidence in top prediction: {top_conf:.1%}")
            else:
                st.info("Selected model does not expose class probabilities.")

# ---------------------------------------------------------------------------
# TAB 2: Batch prediction from an uploaded CSV
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV of bean measurements")
    st.caption(
        "The CSV must contain these columns (any extra columns are ignored): "
        + ", ".join(feature_names)
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read that file as CSV: {e}")
            batch_df = None

        if batch_df is not None:
            missing_cols = [c for c in feature_names if c not in batch_df.columns]
            if missing_cols:
                st.error(f"The uploaded file is missing required columns: {missing_cols}")
            else:
                scaled_batch = scaler.transform(batch_df[feature_names])
                preds = model.predict(scaled_batch)
                batch_df["Predicted_Class"] = label_encoder.inverse_transform(preds)

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(scaled_batch)
                    batch_df["Prediction_Confidence"] = proba.max(axis=1).round(4)

                st.success(f"Predicted {len(batch_df)} rows.")
                st.dataframe(batch_df, use_container_width=True)

                csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="bean_predictions.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------------------------
# TAB 3: About / model transparency
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("Model Details")
    if metadata:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best Model", metadata.get("best_model_name", "N/A"))
        m2.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0):.2%}")
        m3.metric("Weighted F1", f"{metadata.get('weighted_f1', 0):.2%}")
        m4.metric("Overfitting Gap", f"{metadata.get('overfitting_gap', 0):.2%}")

        st.write(f"**Trained at:** {metadata.get('trained_at', 'N/A')}")
        st.write(f"**Class-imbalance technique used:** "
                  f"{'SMOTE oversampling' if metadata.get('used_smote') else 'None needed for winning model'}")
        st.write(f"**Best hyperparameters:** `{metadata.get('best_params', {})}`")
        st.write(f"**Classes the model can predict:** {', '.join(metadata.get('classes', []))}")
    else:
        st.info("No metadata.json found -- run train_pipeline.py to generate it.")

    st.markdown("---")
    st.write(
        "This app was generated as part of an end-to-end supervised machine "
        "learning project: EDA → outlier treatment → feature scaling → "
        "multi-model comparison with cross-validation → class-imbalance "
        "handling → hyperparameter tuning → this deployed Streamlit app. "
        "See `notebooks/Beans_Multiclass_Classification.ipynb` for the full "
        "walkthrough with explanations, and `models/model_comparison.csv` "
        "for the full model comparison table."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center;'>

### 🫘 Dry Bean Classifier using Machine Learning
Built with ❤️ using Streamlit | Developed by nmshah9

</div>
""", unsafe_allow_html=True)
