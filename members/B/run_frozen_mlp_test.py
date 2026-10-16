"""동결된 C1 MLP를 보류 test 세트에서 최종 1회 평가한다.

검증 세트에서 선택한 임계값을 --threshold로 명시해야 한다. 결과 파일이 이미
존재하면 재평가 및 덮어쓰기를 막는다. 학습이나 모델 수정은 수행하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
C1_DIR = ROOT / "members" / "C1"
MODEL_DIR = C1_DIR / "final_model"
TEST_PATH = ROOT / "data" / "final" / "test_final.csv"
METRIC_PATH = ROOT / "results" / "B" / "metrics" / "final_test_mlp_metrics.csv"
PREDICTION_PATH = (
    ROOT / "results" / "B" / "predictions" / "final_test_mlp_predictions.csv"
)


def evaluate(threshold: float) -> None:
    if not 0 < threshold < 1:
        raise ValueError("--threshold는 0과 1 사이여야 합니다.")
    if METRIC_PATH.exists() or PREDICTION_PATH.exists():
        raise FileExistsError(
            "최종 test 결과가 이미 있습니다. 기존 결과를 확인하세요: "
            f"{METRIC_PATH}, {PREDICTION_PATH}"
        )

    with (MODEL_DIR / "model_card.json").open(encoding="utf-8") as stream:
        card = json.load(stream)
    if card["input_dim"] != 26 or card["model_name"] != "MLP_final_seed42":
        raise ValueError("동결 모델 카드가 확정된 26차원 seed 42 모델과 다릅니다.")

    sys.path.insert(0, str(C1_DIR))
    import predict_mlp  # pylint: disable=import-outside-toplevel

    test = pd.read_csv(TEST_PATH)
    expected_rows = card["data"].get("test_rows", 2441)
    if len(test) != expected_rows:
        raise ValueError(f"test 행 수가 {len(test)}입니다. 기대값: {expected_rows}")
    if "Revenue" not in test.columns or test.isna().any().any():
        raise ValueError("test에 Revenue가 없거나 결측치가 있습니다.")
    y_true = test["Revenue"].astype(int).to_numpy()
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Revenue가 이진 값이 아닙니다.")

    artifacts = predict_mlp.load_artifacts(str(MODEL_DIR))
    if len(artifacts["columns"]) != 26:
        raise ValueError("저장된 입력 열 목록이 26차원이 아닙니다.")
    scores = predict_mlp.predict_proba(test, artifacts)
    if len(scores) != len(test) or not np.isfinite(scores).all():
        raise ValueError("test 예측 점수에 누락 또는 비유한 값이 있습니다.")
    if not np.logical_and(scores >= 0, scores <= 1).all():
        raise ValueError("test 예측 점수가 0~1 범위를 벗어났습니다.")

    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    metrics = pd.DataFrame(
        [
            {
                "model": card["model_name"],
                "evaluation": "held-out final test",
                "threshold": threshold,
                "n_test": len(test),
                "n_positive": int(y_true.sum()),
                "precision": precision_score(y_true, predicted, zero_division=0),
                "recall": recall_score(y_true, predicted, zero_division=0),
                "f1": f1_score(y_true, predicted, zero_division=0),
                "roc_auc": roc_auc_score(y_true, scores),
                "average_precision": average_precision_score(y_true, scores),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "model_frozen_at": card["frozen_at"],
            }
        ]
    )
    predictions = pd.DataFrame(
        {
            "row_id": np.arange(len(test)),
            "split": "test",
            "y_true": y_true,
            "y_probability": scores,
            "y_pred": predicted,
            "model_name": card["model_name"],
            "threshold": threshold,
        }
    )

    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRIC_PATH, index=False)
    predictions.to_csv(PREDICTION_PATH, index=False)
    print(metrics.round(6).to_string(index=False))
    print(f"Saved: {METRIC_PATH}")
    print(f"Saved: {PREDICTION_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="validation에서 test 확인 전에 확정한 분류 임계값",
    )
    arguments = parser.parse_args()
    evaluate(arguments.threshold)
