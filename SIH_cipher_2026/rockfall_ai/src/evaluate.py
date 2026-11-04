"""Model evaluation and reporting plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float], np.ndarray]:
    """Return standard binary classification metrics and probability predictions."""
    prediction = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }, probabilities


def plot_confusion(model: Any, x_test: pd.DataFrame, y_test: pd.Series, output: Path) -> None:
    """Save one confusion matrix for the chosen model."""
    fig, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=["Safe", "Rockfall"], ax=axis, cmap="Blues")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def plot_eda(features: pd.DataFrame, target: pd.Series, plots_dir: Path) -> None:
    """Save class distribution, missingness, and numeric-correlation plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 4)); target.value_counts().sort_index().plot.bar(ax=axis, color=["#4c78a8", "#e45756"])
    axis.set_xticklabels(["Safe (0)", "Rockfall (1)"], rotation=0); axis.set_title("Class distribution"); fig.tight_layout(); fig.savefig(plots_dir / "class_distribution.png", dpi=160); plt.close(fig)
    missing = features.isna().mean().sort_values(ascending=False).head(25)
    fig, axis = plt.subplots(figsize=(10, 6)); missing.plot.barh(ax=axis); axis.set_title("Top missing-value rates"); fig.tight_layout(); fig.savefig(plots_dir / "missing_values.png", dpi=160); plt.close(fig)
    numeric = features.select_dtypes(include=np.number)
    if numeric.shape[1] > 1:
        fig, axis = plt.subplots(figsize=(12, 9)); sns.heatmap(numeric.iloc[:, :30].corr(), cmap="coolwarm", center=0, ax=axis)
        axis.set_title("Numeric feature correlations"); fig.tight_layout(); fig.savefig(plots_dir / "correlation_heatmap.png", dpi=160); plt.close(fig)


def plot_feature_importance(model: Any, output: Path, top_n: int = 20) -> None:
    """Save native tree feature importances when exposed by the fitted estimator."""
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = classifier.feature_importances_
    ordered = pd.Series(values, index=names).nlargest(top_n).sort_values()
    fig, axis = plt.subplots(figsize=(10, 7)); ordered.plot.barh(ax=axis, color="#4c78a8"); axis.set_title("Feature importance")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)
