"""B 담당 기준선 모델을 학습하고 공통 평가 자료를 저장한다."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path


def find_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data" / "processed" / "train.csv").exists():
            return candidate
        if (candidate / "data" / "train.csv").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트와 팀 train.csv를 찾을 수 없습니다.")


ROOT = find_project_root()
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = ROOT / "data"
PROCESSED_DIR = (
    DATA_DIR / "processed" if (DATA_DIR / "processed" / "train.csv").exists() else DATA_DIR
)
RAW_DIR = DATA_DIR / "raw" if (DATA_DIR / "raw").exists() else DATA_DIR
RESULT_ROOT = ROOT / "outputs" if (ROOT / "outputs").exists() else ROOT / "results" / "B"
FIGURE_DIR = RESULT_ROOT / "figures"
METRIC_DIR = RESULT_ROOT / "metrics"
PREDICTION_DIR = RESULT_ROOT / "predictions"

RANDOM_STATE = 42
N_SPLITS = 5

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath"
)


def prepare_directories() -> None:
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        FIGURE_DIR,
        METRIC_DIR,
        PREDICTION_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_team_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "data/processed/train.csv과 test.csv가 필요합니다. "
            "팀이 확정한 분할 파일을 복사하세요."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if "Revenue" not in train or "Revenue" not in test:
        raise ValueError("train.csv와 test.csv에 Revenue 컬럼이 필요합니다.")
    if list(train.columns) != list(test.columns):
        raise ValueError("train.csv와 test.csv의 컬럼 순서가 다릅니다.")
    if train.duplicated().any() or test.duplicated().any():
        raise ValueError("팀 분할 파일에 중복 행이 있습니다.")

    train_x = train.drop(columns="Revenue")
    train_y = train["Revenue"].astype(int)
    test_x = test.drop(columns="Revenue")
    test_y = test["Revenue"].astype(int)
    return train_x, train_y, test_x, test_y


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_columns = features.columns.difference(categorical_columns).tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
    threshold: float,
) -> dict[str, float | str]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def save_class_distribution(train_y: pd.Series, test_y: pd.Series) -> pd.DataFrame:
    rows = []
    for split_name, target in (("train", train_y), ("test", test_y)):
        counts = target.value_counts().sort_index()
        for label in (0, 1):
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "split": split_name,
                    "class": "Purchase" if label == 1 else "No purchase",
                    "count": count,
                    "ratio": count / len(target),
                }
            )
    distribution = pd.DataFrame(rows)
    distribution.to_csv(METRIC_DIR / "class_distribution.csv", index=False)
    return distribution


def main(evaluate_test: bool = False) -> None:
    prepare_directories()
    sns.set_theme(style="whitegrid")

    train_x, train_y, test_x, test_y = load_team_splits()
    class_distribution = save_class_distribution(train_y, test_y)
    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    estimators = build_models()

    probabilities_by_model: dict[str, np.ndarray] = {}
    metric_rows = []
    prediction_frames = []

    for model_name, estimator in estimators.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(train_x)),
                ("model", estimator),
            ]
        )
        probabilities = cross_val_predict(
            pipeline,
            train_x,
            train_y,
            cv=cross_validation,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        probabilities_by_model[model_name] = probabilities
        metric_rows.append(
            calculate_metrics(train_y, probabilities, model_name, threshold=0.50)
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(train_x)),
                    "split": "train_oof",
                    "y_true": train_y.to_numpy(),
                    "y_probability": probabilities,
                    "y_pred_0_5": (probabilities >= 0.50).astype(int),
                    "model_name": model_name,
                }
            )
        )

    baseline_metrics = pd.DataFrame(metric_rows).sort_values("f1", ascending=False)
    baseline_metrics.insert(1, "evaluation", "5-fold out-of-fold")
    baseline_metrics.to_csv(METRIC_DIR / "cv_baseline_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        PREDICTION_DIR / "cv_baseline_predictions.csv", index=False
    )

    threshold_rows = []
    for model_name, probabilities in probabilities_by_model.items():
        for threshold in np.arange(0.20, 0.81, 0.05):
            threshold_rows.append(
                calculate_metrics(
                    train_y,
                    probabilities,
                    model_name,
                    threshold=round(float(threshold), 2),
                )
            )
    threshold_results = pd.DataFrame(threshold_rows)
    threshold_results.to_csv(METRIC_DIR / "cv_threshold_analysis.csv", index=False)
    threshold_candidates = (
        threshold_results.sort_values(["model", "f1"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(3)
    )
    threshold_candidates.to_csv(
        METRIC_DIR / "cv_threshold_candidates.csv", index=False
    )

    train_distribution = class_distribution.query("split == 'train'")
    figure, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=train_distribution,
        x="class",
        y="count",
        hue="class",
        palette=["#6B7280", "#2563EB"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Training target class distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Sessions")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "train_class_distribution.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(estimators), figsize=(12, 4))
    for axis, (model_name, probabilities) in zip(
        axes, probabilities_by_model.items()
    ):
        predictions = (probabilities >= 0.50).astype(int)
        matrix = confusion_matrix(train_y, predictions)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(model_name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "cv_confusion_matrices.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model_name, probabilities in probabilities_by_model.items():
        RocCurveDisplay.from_predictions(
            train_y, probabilities, name=model_name, ax=axes[0]
        )
        PrecisionRecallDisplay.from_predictions(
            train_y, probabilities, name=model_name, ax=axes[1]
        )
    axes[0].set_title("ROC curve")
    axes[1].set_title("Precision-Recall curve")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "cv_roc_pr_curves.png", dpi=180)
    plt.close(figure)

    print(f"Team train: {len(train_x):,} rows, {train_x.shape[1]} features")
    print(f"Held-out test: {len(test_x):,} rows (not evaluated)")
    print("\n5-fold out-of-fold metrics at threshold 0.50")
    print(baseline_metrics.round(4).to_string(index=False))
    print("\nTop threshold candidates by F1 (candidates only, not final)")
    print(threshold_candidates.round(4).to_string(index=False))

    if evaluate_test:
        test_metric_rows = []
        test_prediction_frames = []
        for model_name, estimator in build_models().items():
            pipeline = Pipeline(
                [
                    ("preprocessor", build_preprocessor(train_x)),
                    ("model", estimator),
                ]
            )
            pipeline.fit(train_x, train_y)
            probabilities = pipeline.predict_proba(test_x)[:, 1]
            test_metric_rows.append(
                calculate_metrics(test_y, probabilities, model_name, threshold=0.50)
            )
            test_prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_id": np.arange(len(test_x)),
                        "split": "test",
                        "y_true": test_y.to_numpy(),
                        "y_probability": probabilities,
                        "y_pred_0_5": (probabilities >= 0.50).astype(int),
                        "model_name": model_name,
                    }
                )
            )
        pd.DataFrame(test_metric_rows).to_csv(
            METRIC_DIR / "final_test_metrics.csv", index=False
        )
        pd.concat(test_prediction_frames, ignore_index=True).to_csv(
            PREDICTION_DIR / "final_test_predictions.csv", index=False
        )
        print("\nFinal test evaluation was requested and saved.")
    else:
        print("\nThe held-out test set remains untouched.")
    print(f"\nSaved outputs under: {RESULT_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="설정을 확정한 뒤 최종 테스트를 한 번 실행한다.",
    )
    arguments = parser.parse_args()
    main(evaluate_test=arguments.evaluate_test)
