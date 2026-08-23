"""
tuning.py
=========
STEP 8 of the pipeline: Hyperparameter Tuning.

Why tuning happens LAST, only on the top-performing model(s):
    GridSearchCV / RandomizedSearchCV are expensive -- they refit a model
    many times over every parameter combination x every CV fold. Running
    that search across all 8 candidate models from modeling.py would be
    slow and mostly wasted effort. Instead, we first identify the 1-2 best
    performing model families from the plain comparison table, then spend
    the tuning budget only there -- squeezing out real, targeted gains.

Why RandomizedSearchCV is offered alongside GridSearchCV:
    GridSearchCV tries every combination exhaustively, which is fine for a
    small grid (e.g. Random Forest with a handful of options). For a model
    with a larger/continuous parameter space (e.g. SVM's C and gamma),
    RandomizedSearchCV samples a fixed number of combinations instead,
    finding a near-optimal setting in a fraction of the time.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE, CV_FOLDS

# Parameter grids kept intentionally small: this keeps the notebook runnable
# end-to-end in a few minutes on a laptop, while still covering the
# parameters known to matter most for each algorithm.
PARAM_GRIDS = {
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
        "search_type": "grid",
    },
    "SVM": {
        "estimator": SVC(random_state=RANDOM_STATE, probability=True),
        "param_grid": {
            "C": [0.1, 1, 10, 50],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf"],
        },
        "search_type": "random",
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "search_type": "grid",
    },
}


def tune_model(model_name: str, X_train, y_train, cv_folds: int = CV_FOLDS, n_iter: int = 15):
    """
    Run GridSearchCV or RandomizedSearchCV (per PARAM_GRIDS config) for the
    named model and return the best estimator, best params, and best score.

    Why StratifiedKFold is reused here too: the same class-imbalance
    reasoning from modeling.py applies during tuning -- every fold searched
    over must keep realistic class proportions, or the "best" parameters
    chosen could be the ones that happen to overfit an unrepresentative fold.
    """
    if model_name not in PARAM_GRIDS:
        raise ValueError(f"No parameter grid defined for '{model_name}'. "
                          f"Available: {list(PARAM_GRIDS.keys())}")

    cfg = PARAM_GRIDS[model_name]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    if cfg["search_type"] == "grid":
        search = GridSearchCV(
            cfg["estimator"], cfg["param_grid"], cv=skf,
            scoring="f1_weighted", n_jobs=-1, verbose=1,
        )
    else:
        search = RandomizedSearchCV(
            cfg["estimator"], cfg["param_grid"], n_iter=n_iter, cv=skf,
            scoring="f1_weighted", n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
        )

    print(f"Tuning {model_name} using {cfg['search_type']} search ...")
    search.fit(X_train, y_train)

    print(f"Best params for {model_name}: {search.best_params_}")
    print(f"Best CV weighted-F1 for {model_name}: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_
