import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve
)


def plot_roc_curve(y_true, probabilities, label, ax=None):
    """Plot a ROC curve for a model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, probabilities)

    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    return ax


def plot_precision_recall_curve(y_true, probabilities, label, ax=None):
    """Plot a precision-recall curve for a model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities
    )

    ax.plot(recall, precision, linewidth=2, label=label)
    ax.axhline(
        y_true.mean(),
        linestyle="--",
        color="gray",
        label="Baseline cancellation rate"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()

    return ax


def plot_confusion_matrix(y_true, predictions, title):
    """Display a cancellation classification confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Not cancelled", "Cancelled"],
        cmap="Blues",
        values_format=",d",
        ax=ax
    )

    ax.set_title(title)
    plt.tight_layout()

    return ax


def plot_feature_importance(
    importance_df,
    feature_col="feature",
    value_col="relative_importance_pct",
    top_n=20,
    title="Feature Importance"
):
    """Plot a horizontal top-feature importance chart."""
    plot_data = (
        importance_df
        .head(top_n)
        .sort_values(value_col, ascending=True)
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=plot_data,
        x=value_col,
        y=feature_col,
        color="#4C78A8"
    )
    plt.title(title)
    plt.xlabel(value_col.replace("_", " ").title())
    plt.ylabel("Feature")
    plt.tight_layout()
