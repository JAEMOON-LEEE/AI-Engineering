"""동결된 최종 MLP와 B 기준선을 공식 validation에서 통합 평가한다.

C1이 저장한 validation 예측 확률을 읽어 로지스틱 회귀·랜덤포레스트와
동일한 행, 지표, 임계값 기준으로 비교한다. test_final.csv는 읽지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from run_baseline import (
    FIGURE_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    ROOT,
    calculate_metrics,
    prepare_directories,
)


VALIDATION_PATH = ROOT / "data" / "final" / "validation_final.csv"
BASELINE_PREDICTION_PATH = (
    PREDICTION_DIR / "validation_baseline_predictions.csv"
)
MLP_PREDICTION_PATH = (
    ROOT
    / "members"
    / "C1"
    / "final_model"
    / "validation_predictions_mlp.csv"
)

MODEL_ORDER = ["Logistic Regression", "Random Forest", "MLP_final_seed42"]
DISPLAY_NAME = {
    "Logistic Regression": "Logistic Regression",
    "Random Forest": "Random Forest",
    "MLP_final_seed42": "MLP (final seed 42)",
}
REQUIRED_COLUMNS = {
    "row_id",
    "split",
    "y_true",
    "y_probability",
    "model_name",
}


def _validate_prediction_frame(
    frame: pd.DataFrame,
    validation_y: np.ndarray,
    source_path: Path,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{source_path}에 필수 컴럼이 없습니다: {sorted(missing)}")
    if not frame["split"].eq("validation").all():
        raise ValueError(f"{source_path}에 validation 외 split이 있습니다.")
    if frame["y_probability"].isna().any():
        raise ValueError(f"{source_path}에 예측 확률 결측치가 있습니다.")
    if not frame["y_probability"].between(0, 1).all():
        raise ValueError(f"{source_path}의 예측 확률이 0~1 범위를 벗어납니다.")

    checked = []
    expected_ids = np.arange(len(validation_y))
    for model_name, group in frame.groupby("model_name", sort=False):
        group = group.sort_values("row_id").reset_index(drop=True)
        if len(group) != len(validation_y):
            raise ValueError(
                f"{model_name}의 예측 행이 {len(group)}개입니다. "
                f"validation {len(validation_y)}개와 다릅니다."
            )
        if not np.array_equal(group["row_id"].to_numpy(), expected_ids):
            raise ValueError(f"{model_name}의 row_id가 0부터 순서대로 이어지지 않습니다.")
        if not np.array_equal(group["y_true"].astype(int).to_numpy(), validation_y):
            raise ValueError(f"{model_name}의 y_true가 validation_final.csv와 다릅니다.")
        group["y_pred_0_5"] = (group["y_probability"] >= 0.50).astype(int)
        checked.append(group)
    return pd.concat(checked, ignore_index=True)


def load_predictions() -> tuple[pd.DataFrame, np.ndarray]:
    for path in (
        VALIDATION_PATH,
        BASELINE_PREDICTION_PATH,
        MLP_PREDICTION_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")

    validation = pd.read_csv(VALIDATION_PATH)
    validation_y = validation["Revenue"].astype(int).to_numpy()
    baseline = _validate_prediction_frame(
        pd.read_csv(BASELINE_PREDICTION_PATH),
        validation_y,
        BASELINE_PREDICTION_PATH,
    )
    mlp = _validate_prediction_frame(
        pd.read_csv(MLP_PREDICTION_PATH),
        validation_y,
        MLP_PREDICTION_PATH,
    )
    combined = pd.concat([baseline, mlp], ignore_index=True)

    actual_models = set(combined["model_name"])
    if actual_models != set(MODEL_ORDER):
        raise ValueError(
            f"비교 모델이 예상과 다릅니다. "
            f"예상={MODEL_ORDER}, 실제={sorted(actual_models)}"
        )
    combined["model_name"] = pd.Categorical(
        combined["model_name"], categories=MODEL_ORDER, ordered=True
    )
    combined = combined.sort_values(["model_name", "row_id"]).reset_index(drop=True)
    combined["model_name"] = combined["model_name"].astype(str)
    return combined, validation_y


def build_metrics(
    combined: pd.DataFrame,
    validation_y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    threshold_rows = []
    for model_name in MODEL_ORDER:
        probabilities = combined.loc[
            combined["model_name"].eq(model_name), "y_probability"
        ].to_numpy()
        row = calculate_metrics(
            validation_y, probabilities, model_name, threshold=0.50
        )
        row["evaluation"] = "official validation"
        metrics.append(row)
        for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
            threshold_rows.append(
                calculate_metrics(
                    validation_y,
                    probabilities,
                    model_name,
                    threshold=float(threshold),
                )
            )

    metric_frame = pd.DataFrame(metrics)[
        [
            "model",
            "evaluation",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]
    ].sort_values("average_precision", ascending=False)

    threshold_frame = pd.DataFrame(threshold_rows)
    candidate_frame = (
        threshold_frame.sort_values(
            ["model", "f1", "recall"], ascending=[True, False, False]
        )
        .groupby("model", as_index=False)
        .head(3)
        .copy()
    )
    candidate_frame.insert(
        1,
        "candidate_rank_by_f1",
        candidate_frame.groupby("model").cumcount() + 1,
    )
    return metric_frame, threshold_frame, candidate_frame


def save_error_analysis(combined: pd.DataFrame) -> None:
    validation = pd.read_csv(VALIDATION_PATH).reset_index(names="row_id")
    mlp = combined[combined["model_name"].eq("MLP_final_seed42")][
        ["row_id", "y_probability", "y_pred_0_5", "model_name"]
    ]
    analysis = validation.merge(mlp, on="row_id", how="left", validate="one_to_one")
    analysis["error_type"] = np.select(
        [
            analysis["Revenue"].eq(1) & analysis["y_pred_0_5"].eq(0),
            analysis["Revenue"].eq(0) & analysis["y_pred_0_5"].eq(1),
            analysis["Revenue"].eq(1) & analysis["y_pred_0_5"].eq(1),
        ],
        ["false_negative", "false_positive", "true_positive"],
        default="true_negative",
    )
    analysis[analysis["error_type"].isin(["false_negative", "false_positive"])].to_csv(
        PREDICTION_DIR / "validation_mlp_errors_at_0_5.csv", index=False
    )


def save_figures(combined: pd.DataFrame, validation_y: np.ndarray) -> None:
    sns.set_theme(style="whitegrid")
    probabilities = {
        model: combined.loc[
            combined["model_name"].eq(model), "y_probability"
        ].to_numpy()
        for model in MODEL_ORDER
    }

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, model_name in zip(axes, MODEL_ORDER):
        matrix = confusion_matrix(validation_y, probabilities[model_name] >= 0.50)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(DISPLAY_NAME[model_name])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    figure.suptitle("Official validation confusion matrices at threshold 0.50")
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "validation_all_model_confusion_matrices.png", dpi=180
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model_name in MODEL_ORDER:
        RocCurveDisplay.from_predictions(
            validation_y,
            probabilities[model_name],
            name=DISPLAY_NAME[model_name],
            ax=axes[0],
        )
        PrecisionRecallDisplay.from_predictions(
            validation_y,
            probabilities[model_name],
            name=DISPLAY_NAME[model_name],
            ax=axes[1],
        )
    axes[0].set_title("Official validation ROC curve")
    axes[1].set_title("Official validation Precision-Recall curve")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "validation_all_model_roc_pr_curves.png", dpi=180)
    plt.close(figure)


def main() -> None:
    prepare_directories()
    combined, validation_y = load_predictions()
    metrics, thresholds, candidates = build_metrics(combined, validation_y)

    metrics.to_csv(
        METRIC_DIR / "validation_model_comparison_metrics.csv", index=False
    )
    thresholds.to_csv(
        METRIC_DIR / "validation_model_threshold_analysis.csv", index=False
    )
    candidates.to_csv(
        METRIC_DIR / "validation_model_threshold_candidates.csv", index=False
    )
    combined.to_csv(
        PREDICTION_DIR / "validation_all_model_predictions.csv", index=False
    )
    save_error_analysis(combined)
    save_figures(combined, validation_y)

    print("Official validation model comparison at threshold 0.50")
    print(metrics.round(4).to_string(index=False))
    print("\nTop threshold candidates by validation F1 (not final decisions)")
    print(candidates.round(4).to_string(index=False))
    print("\nThe held-out test file was not read.")
    print(f"Saved outputs under: {METRIC_DIR.parent}")


if __name__ == "__main__":
    main()
