"""
config.py
=========
Why this file exists:
    Every notebook / script / Streamlit app in this project needs to agree on
    the SAME file paths, the SAME column names, and the SAME random seed.
    If those live scattered across files, a change in one place (e.g. moving
    the model folder) silently breaks another. Centralising them here means
    there is exactly ONE place to edit when something changes, and it
    guarantees the notebook and the Streamlit app always use identical
    preprocessing settings (same scaler file, same feature order) -- which is
    the #1 cause of "it worked in the notebook but crashes in the app" bugs.
"""

import os

# ---------------------------------------------------------------------------
# Project root -- computed from this file's location so the project works
# regardless of which folder you launch Python from (VS Code, CMD, Streamlit
# Cloud, etc.). This avoids brittle hard-coded absolute paths like
# "C:\Users\Nirav\Desktop\project" that break the moment the folder is moved
# or the project is uploaded to GitHub / Streamlit Cloud.
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Raw dataset location
RAW_DATA_PATH = os.path.join(DATA_DIR, "Dry_Bean_Dataset.xlsx")

# Artifacts produced by training -- consumed later by the Streamlit app.
# Keeping these as separate small files (rather than one big pickle) makes
# each artifact easy to inspect, version, and re-use independently.
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
MODEL_COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
# A fixed RANDOM_STATE makes every split, shuffle, and model fit reproducible.
# Without this, re-running the notebook gives slightly different accuracy
# numbers each time, which makes debugging and comparing models unreliable.
RANDOM_STATE = 42

TARGET_COLUMN = "Class"

# Columns as they exist in the raw file, in a fixed order. The Streamlit app
# will build its input form from this exact list, so the column order used
# for training and the column order used for inference always match.
FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

# Human-readable class names, used to label the Streamlit app's dropdown/help
# text and confusion-matrix plots consistently.
CLASS_NAMES = [
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
]

TEST_SIZE = 0.2
CV_FOLDS = 5
