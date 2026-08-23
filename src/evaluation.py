"""
evaluation.py
=============
STEP 7 of the pipeline: Model Evaluation & Overfitting Check.

Why accuracy alone is not enough for this problem:
    Because of the class imbalance found in EDA, a model could score high
    "accuracy" while performing poorly on the rare BOMBAY class. Precision,
    Recall, and F1-Score computed PER CLASS reveal that kind of hidden
    failure, which overall accuracy hides.

Why the overfitting check compares train vs test accuracy explicitly:
    A model (especially an un-pruned Decision Tree or high-K KNN) can
    memorise the training data and score ~100% there while performing much
    worse on the held-out test set. Explicitly comparing the two numbers is
    the simplest, most direct way to catch that before recommending a model
    for production.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(model, X_train, y_train, X_test, y_test, label_encoder, model_name=""):
    """
    Compute train accuracy, test accuracy, per-class precision/recall/F1,
    weighted F1, and an overfitting flag for a single fitted model.

    Why "weighted" F1 for the single headline number: with 7 imbalanced
    classes, weighted F1 accounts for how many true instances of each class
    exist, giving a fairer single-number summary than macro F1 (which
    treats the 520-sample BOMBAY class as equally important as the
    3500-sample DERMASON class) when the business still cares more about
    overall throughput correctness.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    weighted_f1 = f1_score(y_test, test_pred, average="weighted")

    # Overfitting rule of thumb used throughout this project: a gap greater
    # than 5 percentage points between train and test accuracy is flagged.
    # This threshold is a common practical heuristic, not a hard statistical
    # law -- it is meant to prompt a closer look, not to auto-reject a model.
    overfit_gap = train_acc - test_acc
    is_overfitting = overfit_gap > 0.05

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Overfitting gap: {overfit_gap:.4f} -> "
          f"{'LIKELY OVERFITTING' if is_overfitting else 'OK'}")
    print("\nPer-class classification report (test set):")
    print(classification_report(y_test, test_pred, target_names=label_encoder.classes_))

    return {
        "model_name": model_name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "weighted_f1": weighted_f1,
        "overfit_gap": overfit_gap,
        "is_overfitting": is_overfitting,
    }


def plot_confusion_matrix(model, X_test, y_test, label_encoder, model_name="", save_path=None):
    """
    Plot an annotated confusion matrix for one model.

    Why a confusion matrix specifically (beyond the classification report):
    it shows WHICH classes get confused with which -- e.g. whether SIRA and
    DERMASON (visually similar bean types) are commonly mistaken for each
    other -- which is far more actionable for the business than a single
    accuracy number.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def build_comparison_table(all_results: list) -> pd.DataFrame:
    """
    Build the "Model Comparison Table" required by task 9 of the brief:
    Train Accuracy, Test Accuracy, F1 Score, and an Overfitting Y/N flag
    per model, sorted so the best model is easy to spot.

    Why sort by test accuracy (not train accuracy): train accuracy rewards
    memorisation; test accuracy reflects real generalisation to unseen
    beans, which is what actually matters once this ships as a Streamlit
    app used on new measurements.
    """
    rows = []
    for r in all_results:
        rows.append({
            "Model": r["model_name"],
            "Train Accuracy": round(r["train_accuracy"], 4),
            "Test Accuracy": round(r["test_accuracy"], 4),
            "F1 Score": round(r["weighted_f1"], 4),
            "Overfitting (Y/N)": "Y" if r["is_overfitting"] else "N",
        })
    comparison_df = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    return comparison_df
