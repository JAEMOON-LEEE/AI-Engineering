"""
동결된 최종 MLP로 구매 가능성 점수를 계산하는 모듈 (인계용, 2026-10-14)

필요한 파일: final_model/ 폴더 (mlp_final.keras, scaler.joblib, feature_columns.json)
             -> c1_freeze_model.py가 만든다. 이 모듈은 다른 C1 파일에 의존하지 않는다.

[A·시연] 세션 1건
    import predict_mlp
    art = predict_mlp.load_artifacts()          # 앱 시작 시 한 번만
    score = predict_mlp.predict_session({
        "Administrative": 0, "Administrative_Duration": 0.0, "Informational": 0,
        "Informational_Duration": 0.0, "ProductRelated": 20, "ProductRelated_Duration": 600.0,
        "BounceRates": 0.01, "ExitRates": 0.03, "PageValues": 25.0, "SpecialDay": 0.0,
        "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
        "VisitorType": "Returning_Visitor", "Weekend": False,
    }, art)

[B·평가] 데이터프레임 여러 행 (Revenue 열이 있어도 됨, 무시함)
    proba = predict_mlp.predict_proba(pd.read_csv("data/final/test_final.csv"), art)

왜 1건 입력에 별도 처리가 필요한가
    학습 때는 Month·VisitorType을 drop_first=True로 원핫했다(기준 범주 Aug, New_Visitor가 열에서 빠짐).
    세션 1건에 drop_first=True를 쓰면 그 1건의 값 자체가 '첫 범주'라서 열이 통째로 사라져,
    오류 없이 엉뚱한 확률이 나온다. 그래서 drop_first=False로 원핫한 뒤 학습 열 목록에 맞춰
    reindex한다(없는 열은 0 = 기준 범주). 여러 행일 때도 같은 방법이라 결과가 똑같다.

주의
    출력값은 클래스 가중치로 학습해 실제 구매 확률보다 높게 나온다. 화면에는 '구매 가능성 점수'로 표시 권장.
    Month 값은 학습 데이터 표기 그대로 써야 한다(예: 6월은 "June", 나머지는 "Feb", "Mar" 같은 3글자).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import joblib  # noqa: E402
from tensorflow import keras  # noqa: E402

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model")

RAW_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
    "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]


def load_artifacts(model_dir: str = DEFAULT_DIR) -> dict:
    with open(os.path.join(model_dir, "feature_columns.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return dict(
        model=keras.models.load_model(os.path.join(model_dir, "mlp_final.keras")),
        scaler=joblib.load(os.path.join(model_dir, "scaler.joblib")),
        columns=meta["columns"], categorical=meta["categorical"], categories=meta["categories"],
    )


def _to_weekend_int(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return 1
        if s in ("false", "0", "no"):
            return 0
        raise ValueError(f"Weekend 값을 해석할 수 없음: {v!r} (True/False)")
    return int(bool(v))


def to_features(df: pd.DataFrame, art: dict) -> np.ndarray:
    """원본 열 형식의 데이터프레임 -> 표준화된 26차원 입력."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"입력에 없는 열: {missing}")
    X = df[RAW_COLUMNS].copy()
    for c in art["categorical"]:
        X[c] = X[c].astype(str)
        bad = sorted(set(X[c]) - set(art["categories"][c]))
        if bad:
            raise ValueError(f"{c}에 학습 때 없던 값 {bad}. 가능한 값: {art['categories'][c]}")
    X["Weekend"] = X["Weekend"].map(_to_weekend_int)
    X = pd.get_dummies(X, columns=art["categorical"], drop_first=False)
    X = X.reindex(columns=art["columns"], fill_value=0).astype(float)
    return art["scaler"].transform(X).astype("float32")


def predict_proba(df: pd.DataFrame, art: dict | None = None) -> np.ndarray:
    art = art or load_artifacts()
    return art["model"].predict(to_features(df, art), verbose=0).ravel()


def predict_session(session: dict, art: dict | None = None) -> float:
    return float(predict_proba(pd.DataFrame([session]), art)[0])


if __name__ == "__main__":
    art = load_artifacts()
    here = os.path.dirname(os.path.abspath(__file__))
    va = pd.read_csv(os.path.join(here, "..", "..", "data", "final", "validation_final.csv"))
    row = va.drop(columns=["Revenue"]).iloc[0].to_dict()
    print("검증 세트 첫 세션:", row)
    print(f"구매 가능성 점수: {predict_session(row, art):.4f} (실제 구매: {bool(va['Revenue'].iloc[0])})")
