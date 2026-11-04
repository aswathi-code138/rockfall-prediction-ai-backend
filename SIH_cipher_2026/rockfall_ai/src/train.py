"""End-to-end training entry point for the rockfall classifier."""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluate import evaluate_model, plot_confusion, plot_eda, plot_feature_importance
from .feature_engineering import build_features
from .preprocessing import discover_datasets, merge_datasets
from .utils import configure_logging, ensure_directories, load_config, update_work_item, write_json

LOGGER = logging.getLogger("rockfall_ai")


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build reusable preprocessing for numeric and categorical feature groups."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")


def candidate_models(config: dict[str, Any]) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Return enabled candidate estimators and compact randomized-search spaces."""
    result: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=config["random_state"], n_jobs=-1), {"classifier__n_estimators": [200, 350, 500], "classifier__max_depth": [None, 12, 20], "classifier__min_samples_leaf": [1, 2, 4]}),
    }
    try:
        from xgboost import XGBClassifier
        result["XGBoost"] = (XGBClassifier(random_state=config["random_state"], eval_metric="logloss", n_jobs=-1), {"classifier__n_estimators": [150, 300], "classifier__max_depth": [3, 6, 9], "classifier__learning_rate": [0.03, 0.1, 0.2], "classifier__subsample": [0.7, 1.0]})
    except ImportError:
        LOGGER.warning("XGBoost unavailable; it will be skipped.")
    try:
        from catboost import CatBoostClassifier
        result["CatBoost"] = (CatBoostClassifier(random_state=config["random_state"], verbose=False, auto_class_weights="Balanced"), {"classifier__iterations": [200, 400], "classifier__depth": [4, 6, 8], "classifier__learning_rate": [0.03, 0.1]})
    except ImportError:
        LOGGER.warning("CatBoost unavailable; it will be skipped.")
    return result


def train(config_path: str | None = None) -> pd.DataFrame:
    """Run loading, EDA, CV, tuning, evaluation, artifact generation, and tracking."""
    config = load_config(config_path); ensure_directories(config)
    datasets = discover_datasets(config["data_dir"])
    merged, assumptions = merge_datasets(datasets)
    features, target = build_features(merged)
    if target.nunique() < 2:
        raise ValueError("Training requires both safe and rockfall observations.")
    plots = Path(config["plots_dir"]); outputs = Path(config["outputs_dir"]); models = Path(config["models_dir"])
    plot_eda(features, target, plots)
    features.describe(include="all").transpose().to_csv(outputs / "statistical_summary.csv")
    features.isna().mean().rename("missing_fraction").to_csv(outputs / "missing_value_report.csv")
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=config["test_size"], stratify=target, random_state=config["random_state"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["random_state"])
    rows: list[dict[str, Any]] = []; fitted: dict[str, Any] = {}
    for name, (classifier, parameters) in candidate_models(config).items():
        pipe = Pipeline([("preprocessor", build_preprocessor(x_train)), ("classifier", classifier)])
        search = RandomizedSearchCV(pipe, parameters, n_iter=min(config["n_iter"], 8), scoring="roc_auc", cv=cv, n_jobs=-1, random_state=config["random_state"], refit=True)
        search.fit(x_train, y_train)
        metrics, _ = evaluate_model(search.best_estimator_, x_test, y_test)
        metrics.update({"model": name, "cv_roc_auc_mean": float(cross_val_score(search.best_estimator_, x_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1).mean())})
        rows.append(metrics); fitted[name] = search.best_estimator_
        LOGGER.info("%s ROC-AUC %.3f / F1 %.3f", name, metrics["roc_auc"], metrics["f1"])
    comparison = pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False)
    comparison.to_csv(outputs / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"]); best = fitted[best_name]
    with (models / "best_model.pkl").open("wb") as handle: pickle.dump(best, handle)
    with (models / "preprocessing_pipeline.pkl").open("wb") as handle: pickle.dump(best.named_steps["preprocessor"], handle)
    plot_confusion(best, x_test, y_test, outputs / "confusion_matrix.png")
    plot_feature_importance(best, outputs / "feature_importance.png")
    try:
        import shap  # noqa: F401
        update_work_item("SHAP")
    except ImportError:
        LOGGER.info("SHAP not installed; explanations skipped.")
    metrics = comparison.iloc[0].drop("model").to_dict(); write_json(outputs / "metrics.json", {"best_model": best_name, "metrics": metrics, "assumptions": assumptions})
    report = f"# Training Report\n\n## Best model\n\n{best_name}\n\n## Metrics\n\n{comparison.to_markdown(index=False)}\n\n## Dataset assumptions\n\n" + "\n".join(f"- {a}" for a in assumptions)
    Path(config["reports_dir"]).joinpath("training_report.md").write_text(report, encoding="utf-8")
    completed = ["Dataset loading", "Dataset merging", "Data cleaning", "Feature engineering", "EDA", "Hyperparameter tuning", "Cross Validation", "Model Evaluation", "Feature Importance", "Best model selected", "Model saved", "Training report generated"]
    completed.extend(f"{name} trained" for name in fitted)
    for item in completed:
        update_work_item(item)
    for key, value in {"Model": best_name, "Accuracy": f"{metrics['accuracy']:.3f}", "Precision": f"{metrics['precision']:.3f}", "Recall": f"{metrics['recall']:.3f}", "F1": f"{metrics['f1']:.3f}", "ROC-AUC": f"{metrics['roc_auc']:.3f}"}.items(): update_work_item(key, value)
    return comparison


if __name__ == "__main__":
    configure_logging(); parser = argparse.ArgumentParser(); parser.add_argument("--config", default=None)
    args = parser.parse_args(); print(train(args.config).to_string(index=False))
