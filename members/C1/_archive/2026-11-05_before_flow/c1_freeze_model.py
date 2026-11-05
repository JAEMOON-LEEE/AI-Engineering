"""
C1 - 최종 모델 동결 (2026-10-14)

하는 일
  1) 팀 최종 조건으로 시드 42/43/44를 다시 학습하고, 검증 PR-AUC가 ver2_adam_runs.csv(공식 결과)와
     시드별로 똑같은지 확인한다. 다르면 저장하지 않고 멈춘다(--allow-mismatch로만 무시 가능).
  2) 시드 42(팀 고정 시드) 모델을 시연·테스트용 '동결 모델'로 저장한다.
       final_model/mlp_final.keras        모델 (검증 PR-AUC 최고 에폭 가중치)
       final_model/scaler.joblib          학습 셋 기준 StandardScaler
       final_model/feature_columns.json   입력 열 순서(26개) + 범주 값 목록
       final_model/model_card.json        조건·근거·검증 성적(3시드)·주의사항
  3) B가 요청한 형식으로 검증 예측 확률을 저장한다.
       final_model/validation_predictions_mlp.csv  (row_id, split, y_true, y_probability, model_name)
  4) 저장한 파일을 predict_mlp.py로 다시 불러와, 메모리 모델과 같은 확률이 나오는지와
     '세션 1건씩 넣었을 때'도 같은 확률이 나오는지 확인한다(시연 입력 변환 점검).

최종 조건 (C1 ver2_adam + C2 c2_final_experiment base, 10/14 재현 확인)
  은닉층 (28, 56, 28) ReLU + he_normal, 드롭아웃 0.2(모든 은닉층 뒤), 출력 sigmoid
  이진 교차 엔트로피 + 클래스 가중치 balanced, Adam 학습률 0.01/3, 배치 64,
  최대 30에폭 후 검증 PR-AUC 최고 에폭 가중치로 복원

test_final.csv는 읽지 않는다. 테스트 1회 평가는 B가 임계값 확정 후 predict_mlp.predict_proba()로 한다.

사용법 (members/C1 폴더에서):  python c1_freeze_model.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

import c1_ver2_adam as v2   # 전처리·모델·콜백을 공식 스크립트와 똑같이 쓰기 위해 그대로 가져옴
from c1_ver2_adam import keras, tf  # noqa: E402
import joblib  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

C1_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(C1_DIR, "final_model")
SEEDS = [42, 43, 44]
DEPLOY_SEED = 42
MODEL_NAME = "MLP_final_seed42"
CATEGORICAL = ["Month", "VisitorType"]

FINAL = dict(
    hidden=(28, 56, 28), dropout=0.2, optimizer="Adam", lr=0.01 / 3,
    imbalance="class_weight", batch=64, max_epochs=30, early_stop=0,
)


def load_data():
    """c1_ver2_adam.load_data()와 같은 계산. 다만 scaler와 열 목록을 저장하려고 함께 돌려준다."""
    tr = pd.read_csv(os.path.join(v2.DATA_DIR, "train_clean.csv"))
    va = pd.read_csv(os.path.join(v2.DATA_DIR, "validation_final.csv"))
    X_tr, y_tr = v2.preprocess(tr)
    X_va, y_va = v2.preprocess(va, ref_columns=X_tr.columns)
    sc = StandardScaler()
    return dict(
        X_tr=sc.fit_transform(X_tr.astype(float)).astype("float32"), y_tr=y_tr.to_numpy(),
        X_va=sc.transform(X_va[X_tr.columns].astype(float)).astype("float32"), y_va=y_va.to_numpy(),
        scaler=sc, columns=list(X_tr.columns),
        categories={c: sorted(tr[c].astype(str).unique().tolist()) for c in CATEGORICAL},
        raw_va=va, n_tr=len(tr), n_va=len(va),
    )


def train(seed, d):
    """c1_ver2_adam.run_one()과 같은 순서로 학습하고, 모델 객체를 돌려준다."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=d["y_tr"])
    cw = {0: float(w[0]), 1: float(w[1])}
    cbs = [v2.BestWeights()]
    m = v2.build(FINAL, d["X_tr"].shape[1])
    h = m.fit(d["X_tr"], d["y_tr"], validation_data=(d["X_va"], d["y_va"]),
              epochs=FINAL["max_epochs"], batch_size=FINAL["batch"], class_weight=cw,
              callbacks=cbs, verbose=0).history
    p = m.predict(d["X_va"], verbose=0).ravel()
    pred = (p >= 0.5).astype(int)
    res = dict(
        seed=seed,
        val_pr=float(average_precision_score(d["y_va"], p)),
        recall=float(recall_score(d["y_va"], pred, zero_division=0)),
        precision=float(precision_score(d["y_va"], pred, zero_division=0)),
        f1=float(f1_score(d["y_va"], pred, zero_division=0)),
        best_epoch=int(np.argmax(h["val_pr_auc"])) + 1,
    )
    return m, p, res, cw


def official_val_pr():
    """ver2_adam_runs.csv에서 최종 조건의 시드별 검증 PR-AUC를 읽는다."""
    runs = pd.read_csv(v2.RUNS_CSV)
    hit = runs[runs["cfg"] == v2.key_of(FINAL)]
    return {int(s): float(v) for s, v in zip(hit["seed"], hit["val_pr"])}


def main(allow_mismatch=False):
    d = load_data()
    print(f"[data] train {d['n_tr']}행, validation {d['n_va']}행, 입력 {len(d['columns'])}차원")
    ref = official_val_pr()
    if not ref:
        print("[중단] ver2_adam_runs.csv에서 최종 조건 행을 찾지 못함")
        sys.exit(1)

    results, deploy = [], None
    for s in SEEDS:
        m, p, r, cw = train(s, d)
        same = s in ref and r["val_pr"] == ref[s]
        r["official_val_pr"] = ref.get(s)
        r["match"] = bool(same)
        results.append(r)
        print(f"  seed {s}: val PR-AUC {r['val_pr']:.6f} (공식 {ref.get(s, float('nan')):.6f}) "
              f"best_epoch {r['best_epoch']} -> {'일치' if same else '불일치'}")
        if s == DEPLOY_SEED:
            deploy = (m, p, cw)

    if not all(r["match"] for r in results) and not allow_mismatch:
        print("[중단] 공식 결과와 다름. 데이터·전처리·환경을 확인할 것. (무시하려면 --allow-mismatch)")
        sys.exit(1)

    m, p, cw = deploy
    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, "mlp_final.keras")
    m.save(model_path)
    joblib.dump(d["scaler"], os.path.join(OUT_DIR, "scaler.joblib"))
    with open(os.path.join(OUT_DIR, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"columns": d["columns"], "categorical": CATEGORICAL,
                   "categories": d["categories"]}, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "row_id": np.arange(len(p)), "split": "validation", "y_true": d["y_va"],
        "y_probability": p, "model_name": MODEL_NAME,
    }).to_csv(os.path.join(OUT_DIR, "validation_predictions_mlp.csv"), index=False)

    df = pd.DataFrame(results)
    card = dict(
        model_name=MODEL_NAME, frozen_at=dt.date.today().isoformat(), deploy_seed=DEPLOY_SEED,
        config={k: (list(v) if isinstance(v, tuple) else v) for k, v in FINAL.items()},
        config_note=("은닉층 (28,56,28) ReLU+he_normal, 드롭아웃 0.2 모든 은닉층 뒤, 출력 sigmoid, "
                     "이진 교차 엔트로피, 클래스 가중치 balanced, Adam lr=0.01/3, 배치 64, "
                     "최대 30에폭 후 검증 PR-AUC 최고 에폭 가중치"),
        class_weight=cw, n_params=int(m.count_params()), input_dim=len(d["columns"]),
        data=dict(train="data/final/train_clean.csv", train_rows=d["n_tr"],
                  validation="data/final/validation_final.csv", validation_rows=d["n_va"],
                  test="data/final/test_final.csv (이 스크립트는 읽지 않음)"),
        validation_by_seed=results,
        validation_mean=dict(pr_auc=float(df.val_pr.mean()), pr_auc_std=float(df.val_pr.std()),
                             recall=float(df.recall.mean()), precision=float(df.precision.mean()),
                             f1=float(df.f1.mean())),
        sources=["members/C1/ver2_adam_runs.csv", "members/C2/c2_final_results.csv (base 조건)"],
        notes=[
            "y_probability는 클래스 가중치로 학습해 실제 구매 확률보다 높게 나온다. 순위 판단용 점수로 해석할 것.",
            "임계값 0.5는 기본값일 뿐이며, 최종 임계값은 B가 검증 세트로 정한다(테스트 결과로 고르지 말 것).",
            "테스트 세트는 동결 후 1회만 평가한다.",
            "시연 입력은 predict_mlp.predict_session()을 쓸 것 (세션 1건 원핫 문제 처리됨).",
        ],
        versions=dict(tensorflow=tf.__version__, keras=keras.__version__, numpy=np.__version__,
                      pandas=pd.__version__),
    )
    with open(os.path.join(OUT_DIR, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    # ---- 저장한 파일로 다시 예측해서 확인 ----
    import predict_mlp
    art = predict_mlp.load_artifacts(OUT_DIR)
    p_batch = predict_mlp.predict_proba(d["raw_va"], art)
    p_single = np.array([predict_mlp.predict_session(r, art)
                         for r in d["raw_va"].drop(columns=["Revenue"]).to_dict("records")])
    gap_batch = float(np.max(np.abs(p_batch - p)))
    gap_single = float(np.max(np.abs(p_single - p)))
    print(f"[확인] 저장 모델 일괄 예측 vs 메모리 모델: 최대 차이 {gap_batch:.2e}")
    print(f"[확인] 세션 1건씩 예측 vs 메모리 모델: 최대 차이 {gap_single:.2e}")
    if max(gap_batch, gap_single) > 1e-5:
        print("[경고] 저장·불러오기 또는 1건 변환에서 확률이 달라짐. 인계 전에 확인 필요.")
        sys.exit(1)

    print(f"\n[완료] 3시드 검증 PR-AUC {df.val_pr.mean():.4f}±{df.val_pr.std():.4f}, "
          f"동결 모델(seed {DEPLOY_SEED}) {results[SEEDS.index(DEPLOY_SEED)]['val_pr']:.4f}")
    print(f"저장 위치: {OUT_DIR}")
    for fn in sorted(os.listdir(OUT_DIR)):
        print("  -", fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-mismatch", action="store_true")
    main(ap.parse_args().allow_mismatch)
