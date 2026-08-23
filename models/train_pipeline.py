"""
train_pipeline.py
==================
THE single end-to-end script that runs every step of this project in order
and produces every artifact the Streamlit app (app.py) needs to run
successfully on the first try:
    models/scaler.joblib
    models/label_encoder.joblib
    models/best_model.joblib
    models/feature_names.json
    models/model_comparison.csv
    models/metadata.json

Why this script exists separately from the notebook:
    The notebook (notebooks/Beans_Multiclass_Classification.ipynb) is for
    HUMAN reading, explanation, and exploration -- it contains plots,
    printed EDA commentary, and the "why" behind every step. This script is
    for MACHINE execution: run it once (`python train_pipeline.py`) and it
    reproducibly regenerates every artifact the app needs, without a human
    needing to click through notebook cells in order. Both call the exact
    same functions in src/, so they can never drift out of sync with each
    other.

Run this BEFORE launching the Streamlit app for the first time:
    python train_pipeline.py
    streamlit run app.py
"""

import json
import time
from datetime import datetime

import joblib

from src.config import (
    MODELS_DIR, BEST_MODEL_PATH, MODEL_COMPARISON_PATH, METADATA_PATH,
    CV_FOLDS,
)
from src.data_loader import load_raw_data, validate_schema, basic_overview
from src.preprocessing import full_preprocessing_pipeline
from src.imbalance import apply_smote, compute_class_weights
from src.modeling import (
    get_candidate_models, train_and_cross_validate_all, enable_probability_if_svm,
)
from src.evaluation import evaluate_model, build_comparison_table
from src.tuning import tune_model

import os


def main():
    start_time = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Import and Load the Data
    # ------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# STEP 1: LOAD DATA")
    print("#" * 80)
    df = load_raw_data()
    validate_schema(df)
    basic_overview(df)

    # ------------------------------------------------------------------
    # STEP 2: EDA is intentionally NOT re-run here.
    # Why: EDA produces plots for human review, not artifacts the model or
    # app need to function. It lives in the notebook where it can be viewed.
    # This script focuses only on what's needed to reproducibly train and
    # deploy the model.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # STEPS 3 & 4: Missing values, outliers, scaling, encoding, split
    # ------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# STEPS 3-4: PREPROCESSING")
    print("#" * 80)
    prep = full_preprocessing_pipeline(df)
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    label_encoder = prep["label_encoder"]

    # ------------------------------------------------------------------
    # STEP 5: Model Building -- try multiple classifiers with CV
    # ------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# STEP 5: BASELINE MODEL COMPARISON (with cross-validation)")
    print("#" * 80)
    baseline_models = get_candidate_models(class_weight=None)
    fitted_baseline, cv_results = train_and_cross_validate_all(
        baseline_models, X_train, y_train, cv_folds=CV_FOLDS
    )

    baseline_eval = []
    for name, model in fitted_baseline.items():
        result = evaluate_model(model, X_train, y_train, X_test, y_test, label_encoder, name)
        result["cv_mean_accuracy"] = cv_results[name]["cv_mean_accuracy"]
        baseline_eval.append(result)

    baseline_table = build_comparison_table(baseline_eval)
    print("\nBASELINE MODEL COMPARISON TABLE")
    print(baseline_table.to_string(index=False))

    # ------------------------------------------------------------------
    # STEP 6: Handling Class Imbalance
    # Why only the top 3 baseline models are re-tried here (not all 8):
    # SMOTE roughly doubles the training set size, so refitting every model
    # would roughly double total training time for little extra insight --
    # the models that were weakest on raw data rarely become the winner
    # after imbalance treatment.
    # ------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# STEP 6: CLASS IMBALANCE HANDLING")
    print("#" * 80)
    top3_names = baseline_table["Model"].head(3).tolist()
    print(f"Re-training top 3 baseline models on SMOTE-resampled data: {top3_names}")

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    class_weights = compute_class_weights(y_train)
    print(f"Computed balanced class weights: {class_weights}")

    imbalance_eval = []
    all_candidates = get_candidate_models(class_weight=None)
    for name in top3_names:
        model = all_candidates[name]
        model.fit(X_train_smote, y_train_smote)
        result = evaluate_model(
            model, X_train_smote, y_train_smote, X_test, y_test, label_encoder,
            model_name=f"{name} + SMOTE"
        )
        imbalance_eval.append(result)

    # ------------------------------------------------------------------
    # STEP 7: Model Evaluation & Overfitting Check
    # Combine baseline + SMOTE results to pick the single best-performing,
    # least-overfit model to carry forward into hyperparameter tuning.
    # ------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# STEP 7: COMBINED EVALUATION -- SELECTING BEST CANDIDATE")
    print("#" * 80)
    combined_eval = baseline_eval + imbalance_eval
    combined_table = build_comparison_table(combined_eval)
    print(combined_table.to_string(index=False))

    best_row = combined_table.iloc[0]
    best_candidate_name = best_row["Model"]
    used_smote = "+ SMOTE" in best_candidate_name
    base_model_key = best_candidate_name.replace(" + SMOTE", "")
    print(f"\nSelected best candidate for tuning: '{best_candidate_name}' "
          f"(test accuracy={best_row['Test Accuracy']}, F1={best_row['F1 Score']})")

    # ------------------------------------------------------------------
    # STEP 8: Hyperparameter Tuning (only on the winning model family)
    # ------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# STEP 8: HYPERPARAMETER TUNING")
    print("#" * 80)
    tune_X = X_train_smote if used_smote else X_train
    tune_y = y_train_smote if used_smote else y_train

    tunable = {"Random Forest", "SVM", "Decision Tree"}
    if base_model_key in tunable:
        best_model, best_params, best_cv_score = tune_model(
            base_model_key, tune_X, tune_y, cv_folds=3, n_iter=12
        )
        tuning_note = f"Tuned via search over {base_model_key} parameter grid."
    else:
        # Some model families (e.g. Naive Bayes, Logistic Regression's
        # default solver) have little/no meaningful hyperparameter surface
        # for this problem, so we simply keep the already-fitted candidate
        # instead of forcing a pointless search.
        source_models = fitted_baseline if not used_smote else {
            **fitted_baseline,
            **{f"{n} + SMOTE": m for n, m in
               zip(top3_names, [all_candidates[n] for n in top3_names])}
        }
        best_model = source_models.get(best_candidate_name) or source_models.get(base_model_key)
        best_params = getattr(best_model, "get_params", lambda: {})()
        tuning_note = f"No tuning grid defined for '{base_model_key}'; kept best baseline fit."
        print(tuning_note)

    # Refit the tuned model on its training data (search already does this
    # internally for GridSearchCV/RandomizedSearchCV via refit=True, but we
    # fit explicitly here too for the no-tuning-grid fallback branch).
    best_model.fit(tune_X, tune_y)

    # If the winning model is an SVM, rebuild + refit WITH probability
    # estimates enabled, since the Streamlit app displays prediction
    # confidence (see modeling.enable_probability_if_svm's docstring).
    best_model = enable_probability_if_svm(
        best_model, class_weight=(class_weights if not used_smote else None)
    )
    best_model.fit(tune_X, tune_y)

    final_result = evaluate_model(
        best_model, tune_X, tune_y, X_test, y_test, label_encoder,
        model_name=f"FINAL TUNED: {best_candidate_name}"
    )

    # ------------------------------------------------------------------
    # STEP 9: Final Model Comparison Table
    # ------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# STEP 9: FINAL MODEL COMPARISON TABLE")
    print("#" * 80)
    final_eval = baseline_eval + imbalance_eval + [final_result]
    final_table = build_comparison_table(final_eval)
    print(final_table.to_string(index=False))
    final_table.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\nSaved comparison table -> {MODEL_COMPARISON_PATH}")

    # ------------------------------------------------------------------
    # Persist the final model + run metadata
    # (scaler, label_encoder, feature_names.json were already saved inside
    # full_preprocessing_pipeline / preprocessing.py)
    # ------------------------------------------------------------------
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Saved final model -> {BEST_MODEL_PATH}")

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model_name": best_candidate_name,
        "used_smote": used_smote,
        "test_accuracy": float(final_result["test_accuracy"]),
        "train_accuracy": float(final_result["train_accuracy"]),
        "weighted_f1": float(final_result["weighted_f1"]),
        "overfitting_gap": float(final_result["overfit_gap"]),
        "best_params": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                         for k, v in (best_params or {}).items()},
        "classes": list(label_encoder.classes_),
        "training_seconds": round(time.time() - start_time, 1),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata -> {METADATA_PATH}")

    print(f"\nTotal pipeline runtime: {metadata['training_seconds']} seconds")
    print("\nAll artifacts are ready. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
