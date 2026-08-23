"""
preprocessing.py
=================
STEP 3 (Missing Values & Outlier Treatment) and STEP 4 (Feature Engineering
& Preprocessing) of the pipeline.

Why these steps live in ONE shared module (and why that matters for the
Streamlit app):
    The single biggest cause of a Streamlit app failing after deployment is
    "training/inference skew" -- the app scales or encodes a new input
    slightly differently than the data used to train the model. By writing
    the scaler-fit / label-encode / column-order logic exactly once here,
    and importing it in both train_pipeline.py and app.py, that class of bug
    is structurally impossible: the app is guaranteed to preprocess a new
    bean measurement exactly as the training data was preprocessed.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import joblib

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    FEATURE_NAMES_PATH,
)


# ---------------------------------------------------------------------------
# STEP 3: Missing values & outliers
# ---------------------------------------------------------------------------
def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Count nulls per column.

    Why: sklearn models cannot fit on NaNs. Even if this particular dataset
    is clean (sensor-derived, camera-measured data usually is), this check
    must run every time so that a future re-export of the data with missing
    rows is caught immediately instead of crashing deep inside model.fit().
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print("Missing values detected:")
        print(missing)
    return missing


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute any missing numeric values with the column median.

    Why median (not mean): physical measurements like Area/Perimeter can be
    right-skewed (a few very large beans), and the median is more robust to
    that skew than the mean, so imputed values won't be pulled toward
    outliers.
    """
    df = df.copy()
    numeric_cols = df[FEATURE_COLUMNS].columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def detect_outliers_iqr(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Report the number of outliers per feature using the IQR rule
    (outside Q1 - 1.5*IQR to Q3 + 1.5*IQR).

    Why IQR and not Z-score: the IQR method does not assume a normal
    distribution, which is safer here since several features (Area,
    Perimeter) are right-skewed rather than normally distributed.
    """
    columns = columns or FEATURE_COLUMNS
    report = {}
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        report[col] = n_outliers
    report_df = pd.DataFrame.from_dict(report, orient="index", columns=["outlier_count"])
    report_df["outlier_pct"] = (report_df["outlier_count"] / len(df) * 100).round(2)
    return report_df.sort_values("outlier_count", ascending=False)


def cap_outliers_iqr(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Cap (winsorize) outliers to the IQR fence instead of deleting rows.

    Why capping instead of dropping rows:
        This is a multiclass problem with an already-rare class (BOMBAY has
        only ~520 samples). Deleting "outlier" rows risks disproportionately
        removing legitimate large/small beans of the minority class, making
        imbalance worse. Capping keeps every row (and its class label) while
        still limiting the influence of extreme values on distance-based
        models like KNN and SVM.
    """
    df = df.copy()
    columns = columns or FEATURE_COLUMNS
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ---------------------------------------------------------------------------
# STEP 4: Feature engineering & preprocessing
# ---------------------------------------------------------------------------
def check_skewness(df: pd.DataFrame, columns=None, threshold: float = 1.0) -> pd.Series:
    """
    Report skewness per feature; flag any above the threshold.

    Why: highly skewed features can dominate distance-based models (KNN,
    SVM) and slow convergence for gradient-based models (Logistic
    Regression). We check this explicitly rather than transforming blindly,
    since StandardScaler + robust trees (Random Forest) are often resilient
    enough that a log-transform isn't always necessary.
    """
    columns = columns or FEATURE_COLUMNS
    skew = df[columns].skew().sort_values(key=np.abs, ascending=False)
    flagged = skew[skew.abs() > threshold]
    print("Skewness per feature (sorted by magnitude):")
    print(skew)
    if not flagged.empty:
        print(f"\nFeatures above |skew| > {threshold}: {list(flagged.index)}")
    return skew


def encode_target(df: pd.DataFrame, fit: bool = True, encoder: LabelEncoder = None):
    """
    Encode the string class labels (e.g. 'SEKER') into integers the model
    can consume, and persist the encoder to disk.

    Why persist the encoder: the Streamlit app needs to convert a model's
    integer prediction (e.g. 4) back into a human-readable class name (e.g.
    'SEKER'). Saving the *same* fitted LabelEncoder used at training time
    guarantees the app's index-to-label mapping never drifts out of sync
    with the model.
    """
    if fit:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(df[TARGET_COLUMN])
        joblib.dump(encoder, LABEL_ENCODER_PATH)
    else:
        y_encoded = encoder.transform(df[TARGET_COLUMN])
    return y_encoded, encoder


def split_data(df: pd.DataFrame, y_encoded: np.ndarray):
    """
    Stratified train/test split.

    Why stratified: with class sizes ranging from ~520 (BOMBAY) to ~3500
    (DERMASON), a plain random split risks the test set ending up with too
    few (or zero) samples of the rarest class, making the reported metrics
    for that class unreliable. `stratify=y_encoded` forces the same class
    proportions in both the train and test sets as in the full dataset.
    """
    X = df[FEATURE_COLUMNS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit a StandardScaler on the TRAINING data only, then transform both
    train and test, and persist the fitted scaler.

    Why fit only on training data:
        Fitting the scaler on the full dataset (including test data) leaks
        information about the test set's distribution into training --
        "data leakage" -- which makes the reported test accuracy overly
        optimistic and not representative of real-world performance on
        genuinely unseen beans.

    Why persist the scaler:
        The Streamlit app receives ONE new bean's measurements at a time. It
        must scale that single row using the exact same mean/std learned
        from training data -- not re-fit a scaler on one row, which would be
        meaningless. Loading this saved `scaler.joblib` in the app guarantees
        that consistency.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # Save the exact feature order/names used to fit the scaler, so the
    # Streamlit app can build its input DataFrame with matching columns
    # in the matching order -- a common source of silent bugs otherwise.
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(list(X_train.columns), f)

    return X_train_scaled, X_test_scaled, scaler


def full_preprocessing_pipeline(df: pd.DataFrame):
    """
    Convenience wrapper chaining steps 3 and 4 end-to-end, exactly in the
    order the notebook walks through them. Returns everything downstream
    steps (modelling, imbalance handling, evaluation) need.
    """
    check_missing_values(df)
    df = handle_missing_values(df)
    df = cap_outliers_iqr(df)
    check_skewness(df)

    y_encoded, encoder = encode_target(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df, y_encoded)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": encoder,
        "feature_columns": list(X_train.columns),
    }
