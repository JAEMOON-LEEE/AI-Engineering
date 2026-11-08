"""
10단계 동결 확인 전용. 학습하지 않는다.

members/C1/final_model 을 predict_mlp.load_artifacts(폴더)로 불러
검증 6행(구매 3·미구매 3)의 1건 예측과 전체 예측이 같은지,
validation_predictions_mlp.csv 형식·PR-AUC를 확인한다.

사용법: python step10_check.py
"""
from __future__ import annotations

import math
import os
import sys

import pandas as pd
from sklearn.metrics import average_precision_score

import flow_core as fc

FLOW_DIR = fc.FLOW_DIR
C1_DIR = os.path.abspath(os.path.join(FLOW_DIR, ".."))
OUT_DIR = os.path.join(FLOW_DIR, "..", "final_model")
PRED_CSV = os.path.join(OUT_DIR, "validation_predictions_mlp.csv")
EXPECTED_N = 1221
EXPECTED_COLS = ["row_id", "split", "y_true", "y_probability", "model_name"]
EXPECTED_PR = 0.701766


def same6(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-7)


def pick_rows(va):
    y = va["Revenue"].astype(bool)
    buy = va.index[y].tolist()[:3]
    nobuy = va.index[~y].tolist()[:3]
    if len(buy) < 3 or len(nobuy) < 3:
        raise SystemExit("검증 셋에서 구매/미구매 3행을 고르지 못함")
    return [("구매", i) for i in buy] + [("미구매", i) for i in nobuy]


def main():
    if C1_DIR not in sys.path:
        sys.path.insert(0, C1_DIR)
    import predict_mlp

    print("\n=== 10단계 · 동결 확인 ===")
    print(f"모델 폴더: {OUT_DIR}")
    print("학습하지 않음. predict_mlp.load_artifacts(폴더)만 사용.")
    art = predict_mlp.load_artifacts(OUT_DIR)

    va = pd.read_csv(os.path.join(fc.DATA_DIR, "validation_final.csv"))
    print(f"validation_final.csv  {len(va)}행")
    batch = predict_mlp.predict_proba(va, art)
    rows = pick_rows(va)
    print("행  구분  1건확률  전체예측  확인")
    all_ok = True
    for kind, i in rows:
        session = va.drop(columns=["Revenue"]).iloc[i].to_dict()
        one = float(predict_mlp.predict_session(session, art))
        many = float(batch[i])
        ok = same6(one, many)
        all_ok = all_ok and ok
        print(f"{i:3d}  {kind}  {one:.6f}  {many:.6f}  {'일치' if ok else '불일치'}")
    print("1건 vs 전체  " + ("일치" if all_ok else "불일치"))

    pred = pd.read_csv(PRED_CSV)
    n_ok = len(pred) == EXPECTED_N
    cols = list(pred.columns)
    cols_ok = cols == EXPECTED_COLS
    pr = float(average_precision_score(pred["y_true"], pred["y_probability"]))
    pr_ok = same6(pr, EXPECTED_PR)
    print("[확인] validation_predictions_mlp.csv")
    print(f"  행 수 이번 {len(pred)}  기준 {EXPECTED_N}  {'일치' if n_ok else '불일치'}")
    print(f"  열 이름 이번 {cols}  기준 {EXPECTED_COLS}  {'일치' if cols_ok else '불일치'}")
    print(f"  PR-AUC 이번 {pr:.6f}  기준 {EXPECTED_PR:.6f}  {'일치' if pr_ok else '불일치'}")
    ok = all_ok and n_ok and cols_ok and pr_ok
    print("일치" if ok else "불일치")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
