"""
modeling.py
===========
STEP 5 of the pipeline: Model Building -- Try Multiple Classifiers.

Why train several different algorithm families instead of picking one:
    No single algorithm is best for every dataset. Trying a spread of model
    families -- a linear model (Logistic Regression), tree-based models
    (Decision Tree, Random Forest), a distance-based model (KNN), a margin-
    based model (SVM), an ensemble (Gradient Boosting), and a probabilistic
    model (Naive Bayes) -- gives an honest, evidence-based answer to "which
    approach actually fits THIS bean-shape data best", rather than assuming
    it upfront.

Why cross-validation is used on top of a single train/test split:
    A single split's accuracy can vary just by chance depending on which
    rows landed in the test set. Stratified K-Fold cross-validation trains
    and evaluates the model K times on different folds and averages the
    result, giving a far more reliable estimate of how the model will
    perform on new, unseen beans.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import RANDOM_STATE, CV_FOLDS


def get_candidate_models(class_weight=None):
    """
    Build the dictionary of candidate models covering every algorithm family
    named in the task brief.

    Why class_weight is a parameter here: it lets the SAME model-building
    function be reused both for a "plain" run and for a "balanced" run
    (see imbalance.py), instead of duplicating this whole dictionary twice.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight=class_weight
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight=class_weight, max_depth=15
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE, class_weight=class_weight,
            max_depth=20, n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(
            # probability=False during model search: Platt-scaling probability
            # estimates require an expensive internal 5-fold CV inside every
            # single SVM fit. We only pay that cost once, later, for the
            # final chosen model (see train_pipeline.py) -- not for all 8
            # candidate models x 5 CV folds during comparison.
            random_state=RANDOM_STATE, class_weight=class_weight, probability=False
        ),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=150),
        "Gradient Boosting (Ensemble)": GradientBoostingClassifier(
            random_state=RANDOM_STATE, n_estimators=150, max_depth=3
        ),
    }
    return models


def enable_probability_if_svm(model, class_weight=None):
    """
    If the winning model is an SVM, rebuild it with probability=True and
    refit is left to the caller. Every other model already supports
    predict_proba natively, so this is a no-op for them.

    Why this exists: the Streamlit app shows the model's confidence for its
    prediction. SVC only exposes predict_proba when trained with
    probability=True, which we deliberately skipped during the fast model
    comparison stage (see get_candidate_models). This function pays that
    one-time cost only for the single model we actually deploy.
    """
    if isinstance(model, SVC):
        return SVC(
            C=model.C, kernel=model.kernel, gamma=model.gamma,
            random_state=RANDOM_STATE, class_weight=class_weight, probability=True,
        )
    return model


def build_voting_ensemble(class_weight=None):
    """
    A soft-voting ensemble combining Random Forest, SVM, and Logistic
    Regression predictions.

    Why these three specifically: they represent three different learning
    biases (tree splits, margin maximisation, linear decision boundary).
    Combining diverse, individually-decent models tends to cancel out each
    model's individual mistakes better than combining near-identical models.
    """
    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, class_weight=class_weight, n_jobs=-1
    )
    svm = SVC(probability=True, random_state=RANDOM_STATE, class_weight=class_weight)
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight=class_weight)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("lr", lr)],
        voting="soft",
    )
    return ensemble


def cross_validate_model(model, X_train, y_train, cv_folds: int = CV_FOLDS):
    """
    Run Stratified K-Fold cross-validation and return the fold scores.

    Why StratifiedKFold specifically (not plain KFold): it preserves each
    class's proportion within every fold, which matters just as much during
    cross-validation as it did during the original train/test split, given
    the class imbalance identified during EDA.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    return scores


def train_and_cross_validate_all(models: dict, X_train, y_train, cv_folds: int = CV_FOLDS):
    """
    Fit every candidate model on the full training set AND report its
    cross-validation score, returning both the fitted models and a summary
    table of CV performance.

    Why fit on the full training set in addition to CV: cross_val_score
    trains K temporary copies internally and discards them. We separately
    fit one final model on all of X_train so we have an actual usable,
    persisted model afterward -- CV is for evaluation, not for producing the
    artifact we deploy.
    """
    results = {}
    fitted_models = {}
    for name, model in models.items():
        print(f"Training & cross-validating: {name} ...")
        cv_scores = cross_validate_model(model, X_train, y_train, cv_folds)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        results[name] = {
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
        }
        print(f"  CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    return fitted_models, results
