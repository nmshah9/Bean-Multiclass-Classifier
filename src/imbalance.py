"""
imbalance.py
============
STEP 6 of the pipeline: Handling Class Imbalance.

Why this step exists as its own module (and why it runs AFTER the
train/test split, not before):
    EDA showed BOMBAY has ~520 samples vs DERMASON's ~3500 -- a ~6.8x
    imbalance. A model can reach high overall accuracy by mostly ignoring
    the rare classes, which is exactly the kind of failure this business
    problem cannot tolerate (misclassifying a bean type has a real quality-
    control cost). Resampling must be applied ONLY to the training data,
    never the test data, or the reported metrics would be evaluated on
    synthetic data instead of real, unseen beans -- an easy way to fool
    yourself into thinking a model performs better than it really does.
"""

from collections import Counter

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.config import RANDOM_STATE


def apply_smote(X_train, y_train):
    """
    SMOTE (Synthetic Minority Over-sampling Technique): generates new,
    synthetic minority-class samples by interpolating between existing
    minority-class neighbours, rather than just duplicating rows.

    Why SMOTE over plain duplication: duplicating rows can cause a model to
    overfit to those exact repeated points. SMOTE instead creates plausible
    new points in feature space, giving the model more varied examples of
    what a BOMBAY bean's measurements can look like.
    """
    print(f"Before SMOTE: {Counter(y_train)}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {Counter(y_resampled)}")
    return X_resampled, y_resampled


def apply_random_oversampling(X_train, y_train):
    """
    Randomly duplicates minority-class rows until all classes are balanced.

    Why include this alongside SMOTE: it's the simplest possible baseline
    for imbalance treatment. Comparing it against SMOTE tells us whether the
    more sophisticated synthetic-sample approach is actually worth the extra
    complexity for this dataset, or whether simple duplication is enough.
    """
    print(f"Before Random Oversampling: {Counter(y_train)}")
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    print(f"After Random Oversampling:  {Counter(y_resampled)}")
    return X_resampled, y_resampled


def apply_random_undersampling(X_train, y_train):
    """
    Randomly removes majority-class rows until all classes match the
    smallest class's count.

    Why this is included but expected to underperform here: undersampling
    DERMASON down to ~520 rows would throw away ~3000 perfectly good
    training examples, which is wasteful when we have plenty of data.
    It's included for a fair, complete comparison against over-sampling
    methods, matching the task brief's explicit ask for both approaches.
    """
    print(f"Before Random Undersampling: {Counter(y_train)}")
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    print(f"After Random Undersampling:  {Counter(y_resampled)}")
    return X_resampled, y_resampled


def compute_class_weights(y_train):
    """
    Compute 'balanced' class weights for use directly inside a classifier's
    class_weight parameter, instead of resampling the data at all.

    Why this is often the best option: unlike SMOTE/oversampling, class
    weighting doesn't change the training set size or add synthetic points --
    it simply tells the loss function to penalise mistakes on rare classes
    more heavily. This is usually the cheapest and most robust of the three
    imbalance strategies, which is why models.py exposes a
    class_weight='balanced' variant for every classifier that supports it.
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))
