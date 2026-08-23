"""
eda.py
======
STEP 2 of the pipeline: Exploratory Data Analysis (EDA).

Why this step matters:
    EDA is where you form hypotheses about what the model will find easy or
    hard. For a bean-classification problem specifically, EDA answers three
    business-critical questions before a single model is trained:
        1. Are the physical measurements (area, perimeter, shape factors)
           actually different across bean types? (feature usefulness)
        2. Is one bean type rare in the data (e.g. Bombay beans), which would
           mean a naive model could get "high accuracy" while completely
           failing to ever detect that type? (class imbalance)
        3. Are any two features almost perfectly correlated (e.g. Area and
           ConvexArea), which would mean we are feeding the model redundant
           information? (multicollinearity)
    Skipping EDA means these issues only surface AFTER a model is trained,
    when they are far more expensive to diagnose.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FEATURE_COLUMNS, TARGET_COLUMN


def plot_class_distribution(df, save_path=None):
    """
    Bar chart of how many samples exist per bean class.

    Why: directly reveals class imbalance. In this dataset, BOMBAY beans
    are far rarer than DERMASON beans -- a model evaluated only on overall
    accuracy could ignore BOMBAY entirely and still look "good", which is
    why this project later applies SMOTE / class weighting (see imbalance.py).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df[TARGET_COLUMN].value_counts().index
    sns.countplot(data=df, x=TARGET_COLUMN, order=order, ax=ax, hue=TARGET_COLUMN,
                   palette="viridis", legend=False)
    ax.set_title("Class Distribution of Bean Types")
    ax.set_xlabel("Bean Class")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_feature_histograms(df, save_path=None):
    """
    Histogram grid, one per numeric feature.

    Why: shows the shape of each feature's distribution (normal, skewed,
    bimodal). Heavily skewed features (common in area/perimeter-type
    measurements) are candidates for scaling or transformation in the
    preprocessing step.
    """
    n_cols = 4
    n_rows = -(-len(FEATURE_COLUMNS) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLUMNS):
        sns.histplot(df[col], kde=True, ax=axes[i], color="teal")
        axes[i].set_title(col)
    for j in range(len(FEATURE_COLUMNS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_feature_boxplots(df, save_path=None):
    """
    Boxplot grid, one per numeric feature.

    Why: boxplots make outliers immediately visible (points beyond the
    whiskers). This directly feeds into the "Missing Values & Outlier
    Treatment" step -- we look here first to decide whether outlier
    treatment is even necessary before blindly applying a rule.
    """
    n_cols = 4
    n_rows = -(-len(FEATURE_COLUMNS) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLUMNS):
        sns.boxplot(y=df[col], ax=axes[i], color="salmon")
        axes[i].set_title(col)
    for j in range(len(FEATURE_COLUMNS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_correlation_heatmap(df, save_path=None):
    """
    Heatmap of pairwise correlations between numeric features.

    Why: shape-derived features are mathematically related (e.g. Area,
    ConvexArea, Perimeter, EquivDiameter all describe "size"). A heatmap
    exposes near-duplicate features. We don't necessarily drop them here --
    tree-based models tolerate correlated features fine -- but it's
    important context when interpreting model coefficients (e.g. Logistic
    Regression) later.
    """
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = df[FEATURE_COLUMNS].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


def plot_pairplot(df, sample_n=800, save_path=None):
    """
    Pairplot of a small subset of the most informative features, coloured
    by class.

    Why a subset and a sample: a full 16-feature pairplot would render a
    16x16 grid (256 panels) which is unreadable and extremely slow. We pick
    a handful of features that summarise size and shape, and subsample rows
    for speed, while still visualising how well classes separate in feature
    space -- a strong visual signal of how "learnable" this problem is.
    """
    subset_cols = ["Area", "Perimeter", "roundness", "Compactness", TARGET_COLUMN]
    plot_df = df[subset_cols]
    if sample_n and len(plot_df) > sample_n:
        plot_df = plot_df.groupby(TARGET_COLUMN, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, sample_n // df[TARGET_COLUMN].nunique())),
                                random_state=42)
        )
    g = sns.pairplot(plot_df, hue=TARGET_COLUMN, palette="husl", diag_kind="kde", height=2.2)
    g.fig.suptitle("Pairplot of Key Size/Shape Features by Class", y=1.02)
    if save_path:
        g.savefig(save_path, dpi=120)
    return g


def summarize_key_findings(df):
    """
    Print a short, human-readable EDA summary.

    Why: EDA is only useful if its conclusions are written down and acted
    on. This function turns the visual exploration above into a few
    concrete, actionable bullet points -- e.g. "class X is under-represented,
    apply SMOTE" -- which is exactly what task 2 of the brief asks for.
    """
    counts = df[TARGET_COLUMN].value_counts()
    imbalance_ratio = counts.max() / counts.min()

    print("KEY EDA FINDINGS")
    print("-" * 60)
    print(f"1. Dataset has {df.shape[0]} samples across {counts.shape[0]} bean classes.")
    print(f"2. Largest class: {counts.idxmax()} ({counts.max()} samples). "
          f"Smallest class: {counts.idxmin()} ({counts.min()} samples).")
    print(f"3. Imbalance ratio (largest/smallest) = {imbalance_ratio:.1f}x -> "
          f"{'meaningful imbalance, treatment recommended' if imbalance_ratio > 3 else 'roughly balanced'}.")
    print("4. Several size-related features (Area, ConvexArea, Perimeter, "
          "EquivDiameter) are highly correlated -- expected, since they all "
          "describe bean size from different angles.")
    print("5. Shape factors and roundness/compactness show visible separation "
          "across classes in the pairplot, suggesting they carry strong "
          "discriminative signal for the classifier.")
