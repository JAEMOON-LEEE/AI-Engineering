"""공식 분할에서 기준선 모델을 학습하고 공통 평가 자료를 저장한다.

기본 실행은 data/final의 train과 validation만 사용한다. test_final.csv는
--evaluate-test를 명시한 최종 평가에서만 읽는다.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path


def find_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data" / "final" / "train_clean.csv").exists():
            return candidate
    raise FileNotFoundError(
        "data/final/train_clean.csv이 있는 프로젝트 루트를 찾을 수 없습니다."
    )


ROOT = find_project_root()
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FINAL_DIR = ROOT / "data" / "final"
TRAIN_PATH = FINAL_DIR / "train_clean.csv"
VALIDATION_PATH = FINAL_DIR / "validation_final.csv"
TEST_PATH = FINAL_DIR / "test_final.csv"

RESULT_ROOT = ROOT / "outputs" if (ROOT / "outputs").exists() else ROOT / "results" / "B"
FIGURE_DIR = RESULT_ROOT / "figures"
METRIC_DIR = RESULT_ROOT / "metrics"
PREDICTION_DIR = RESULT_ROOT / "predictions"

RANDOM_STATE = 42
CATEGORICAL_COLUMNS = ["Month", "VisitorType"]
EXPECTED_INPUT_DIM = 26

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath"
)


def prepare_directories() -> None:
    for directory in (FIGURE_DIR, METRIC_DIR, PREDICTION_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _read_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"공식 {split_name} 파일이 없습니다: {path}")
    frame = pd.read_csv(path)
    if "Revenue" not in frame.columns:
        raise ValueError(f"{path.name}에 Revenue 컬럼이 없습니다.")
    if frame.duplicated().any():
        raise ValueError(f"{path.name}에 중복 행이 있습니다.")
    if frame.isna().any().any():
        raise ValueError(f"{path.name}에 결측치가 있습니다.")
    return frame


def load_development_splits() -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    """공식 train/validation만 읽고 행 겹침까지 검증한다."""
    train = _read_split(TRAIN_PATH, "train")
    validation = _read_split(VALIDATION_PATH, "validation")
    if list(train.columns) != list(validation.columns):
        raise ValueError("train과 validation의 컬럼 순서가 다릅니다.")

    train_hashes = set(pd.util.hash_pandas_object(train, index=False))
    validation_hashes = set(pd.util.hash_pandas_object(validation, index=False))
    if train_hashes & validation_hashes:
        raise ValueError("train과 validation 사이에 동일한 행이 있습니다.")

    return (
        train.drop(columns="Revenue"),
        train["Revenue"].astype(int),
        validation.drop(columns="Revenue"),
        validation["Revenue"].astype(int),
    )


def preprocess_features(
    train_features: pd.DataFrame,
    evaluation_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C1/C2의 official_v1과 같은 26차원 전처리를 적용한다."""
    train_x = train_features.copy()
    evaluation_x = evaluation_features.copy()
    train_x["Weekend"] = train_x["Weekend"].astype(int)
    evaluation_x["Weekend"] = evaluation_x["Weekend"].astype(int)

    train_x = pd.get_dummies(
        train_x,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )
    evaluation_x = pd.get_dummies(
        evaluation_x,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False,
        dtype=int,
    ).reindex(columns=train_x.columns, fill_value=0)

    if list(train_x.columns) != list(evaluation_x.columns):
        raise AssertionError("전처리 후 train/evaluation 컬럼이 일치하지 않습니다.")
    expected_dim = (
        EXPECTED_INPUT_DIM
        if "PageValues" in train_features.columns
        else EXPECTED_INPUT_DIM - 1
    )
    if train_x.shape[1] != expected_dim:
        raise ValueError(
            f"전처리 입력 차원이 {train_x.shape[1]}입니다. "
            f"현재 특징 집합에서는 {expected_dim}차원을 기대했습니다."
        )
    return train_x, evaluation_x


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


def build_pipeline(estimator: object) -> Pipeline:
    """MLP와 동일하게 모든 26개 입력을 train 기준으로 표준화한다."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def save_class_distribution(
    train_y: pd.Series, validation_y: pd.Series
) -> pd.DataFrame:
    rows = []
    for split_name, target in (("train", train_y), ("validation", validation_y)):
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


def save_validation_figures(
    validation_y: pd.Series,
    probabilities_by_model: dict[str, np.ndarray],
    class_distribution: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    figure, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=class_distribution,
        x="split",
        y="count",
        hue="class",
        palette=["#6B7280", "#2563EB"],
        ax=ax,
    )
    ax.set_title("Target class distribution by development split")
    ax.set_xlabel("")
    ax.set_ylabel("Sessions")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "development_class_distribution.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(probabilities_by_model), figsize=(12, 4))
    for axis, (model_name, probabilities) in zip(
        axes, probabilities_by_model.items()
    ):
        predictions = (probabilities >= 0.50).astype(int)
        matrix = confusion_matrix(validation_y, predictions)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(model_name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "validation_confusion_matrices.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model_name, probabilities in probabilities_by_model.items():
        RocCurveDisplay.from_predictions(
            validation_y, probabilities, name=model_name, ax=axes[0]
        )
        PrecisionRecallDisplay.from_predictions(
            validation_y, probabilities, name=model_name, ax=axes[1]
        )
    axes[0].set_title("Official validation ROC curve")
    axes[1].set_title("Official validation Precision-Recall curve")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "validation_roc_pr_curves.png", dpi=180)
    plt.close(figure)


def evaluate_test_once(
    raw_train_x: pd.DataFrame,
    train_y: pd.Series,
    raw_validation_x: pd.DataFrame,
    validation_y: pd.Series,
) -> None:
    """명시적 플래그가 있을 때만 test를 읽어 최종 0.5 임계값 평가를 수행한다."""
    test = _read_split(TEST_PATH, "test")
    development_x = pd.concat([raw_train_x, raw_validation_x], ignore_index=True)
    development_y = pd.concat([train_y, validation_y], ignore_index=True)
    development_x, test_x = preprocess_features(
        development_x, test.drop(columns="Revenue")
    )
    test_y = test["Revenue"].astype(int)

    metric_rows = []
    prediction_frames = []
    for model_name, estimator in build_models().items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(development_x, development_y)
        probabilities = pipeline.predict_proba(test_x)[:, 1]
        metric_rows.append(
            calculate_metrics(test_y, probabilities, model_name, threshold=0.50)
        )
        prediction_frames.append(
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

    pd.DataFrame(metric_rows).to_csv(
        METRIC_DIR / "final_test_metrics.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        PREDICTION_DIR / "final_test_predictions.csv", index=False
    )
    print("\nFinal test evaluation was explicitly requested and saved.")


def main(evaluate_test: bool = False) -> None:
    prepare_directories()
    raw_train_x, train_y, raw_validation_x, validation_y = load_development_splits()
    train_x, validation_x = preprocess_features(raw_train_x, raw_validation_x)
    class_distribution = save_class_distribution(train_y, validation_y)

    probabilities_by_model: dict[str, np.ndarray] = {}
    metric_rows = []
    prediction_frames = []

    for model_name, estimator in build_models().items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(train_x, train_y)
        probabilities = pipeline.predict_proba(validation_x)[:, 1]
        probabilities_by_model[model_name] = probabilities
        metric_rows.append(
            calculate_metrics(
                validation_y, probabilities, model_name, threshold=0.50
            )
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(validation_x)),
                    "split": "validation",
                    "y_true": validation_y.to_numpy(),
                    "y_probability": probabilities,
                    "y_pred_0_5": (probabilities >= 0.50).astype(int),
                    "model_name": model_name,
                }
            )
        )

    baseline_metrics = pd.DataFrame(metric_rows).sort_values(
        "average_precision", ascending=False
    )
    baseline_metrics.insert(1, "evaluation", "official validation")
    baseline_metrics.to_csv(
        METRIC_DIR / "validation_baseline_metrics.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        PREDICTION_DIR / "validation_baseline_predictions.csv", index=False
    )

    threshold_rows = []
    for model_name, probabilities in probabilities_by_model.items():
        for threshold in np.arange(0.20, 0.81, 0.05):
            threshold_rows.append(
                calculate_metrics(
                    validation_y,
                    probabilities,
                    model_name,
                    threshold=round(float(threshold), 2),
                )
            )
    threshold_results = pd.DataFrame(threshold_rows)
    threshold_results.to_csv(
        METRIC_DIR / "validation_threshold_analysis.csv", index=False
    )
    threshold_candidates = (
        threshold_results.sort_values(["model", "f1"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(3)
    )
    threshold_candidates.to_csv(
        METRIC_DIR / "validation_threshold_candidates.csv", index=False
    )

    save_validation_figures(
        validation_y, probabilities_by_model, class_distribution
    )

    print(f"Official train: {len(train_x):,} rows, {train_x.shape[1]} features")
    print(f"Official validation: {len(validation_x):,} rows")
    print("\nOfficial validation metrics at threshold 0.50")
    print(baseline_metrics.round(4).to_string(index=False))
    print("\nTop validation threshold candidates by F1 (candidates only)")
    print(threshold_candidates.round(4).to_string(index=False))

    if evaluate_test:
        evaluate_test_once(
            raw_train_x, train_y, raw_validation_x, validation_y
        )
    else:
        print("\nThe held-out test file was not read.")
    print(f"\nSaved outputs under: {RESULT_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="모든 설정을 확정한 뒤 test를 한 번만 최종 평가한다.",
    )
    arguments = parser.parse_args()
    main(evaluate_test=arguments.evaluate_test)
