"""
10단계 · 최종 모델 동결 (시드 42).

설정은 step09_epoch.compute()가 고른 값을 그대로 쓴다.
학습 절차는 flow_core.run_one 과 같다. 다만 모델을 저장하므로 복사한 함수를 쓰고,
flow_runs.csv 에는 쓰지 않는다.

저장: members/C1/final_model/
  mlp_final.keras, scaler.joblib, feature_columns.json,
  validation_predictions_mlp.csv, model_card.json

predict_mlp.py 는 건드리지 않는다.
test_final.csv 는 읽지 않는다. 임계값은 정하지 않는다.

사용법: python step10_freeze.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

import flow_core as fc
from step09_epoch import compute as step09_compute

FLOW_DIR = fc.FLOW_DIR
C1_DIR = os.path.abspath(os.path.join(FLOW_DIR, ".."))
OUT_DIR = os.path.join(FLOW_DIR, "..", "final_model")
DEPLOY_SEED = 42
MODEL_NAME = "MLP_final_seed42"
MODEL_VERSION = "56-112-56, Adam 0.01/3, batch 64, dropout 0.2, 2026-11-05"
CATEGORICAL = ["Month", "VisitorType"]


def py(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [py(x) for x in v]
    return v


def same6(a, b):
    return f"{float(a):.6f}" == f"{float(b):.6f}"


def load_raw():
    """범주 목록·세션 1건용. train/validation 만 읽고 test 는 읽지 않는다."""
    tr = pd.read_csv(os.path.join(fc.DATA_DIR, "train_clean.csv"))
    va = pd.read_csv(os.path.join(fc.DATA_DIR, "validation_final.csv"))
    return dict(
        n_tr=len(tr), n_va=len(va),
        raw_va=va,
        categories={c: sorted(tr[c].astype(str).unique().tolist()) for c in CATEGORICAL},
    )


def train_keep_model(cfg, seed):
    """flow_core.run_one 과 같은 학습. CSV 기록은 하지 않고 최고 에폭 모델을 돌려준다."""
    cfg = fc.cfg_of(**cfg)
    _, keras, _ = fc.tf_modules()
    EpochSnapshot, BestWeights = fc._callbacks()
    d = fc.load_data()
    print(f"  [학습] {fc.cfg_line(cfg)} · 시드 {seed}", flush=True)

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    x_fit, y_fit = d["X_tr"], d["y_tr"]
    if cfg["class_weight"] == "under_1to4":
        x_fit, y_fit = fc.under_1to4(x_fit, y_fit, seed)
    cw = fc.class_weight_of(y_fit, cfg["class_weight"])

    snap = EpochSnapshot()
    callbacks = [snap]
    if cfg["early_stop"]:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=cfg["early_stop"], restore_best_weights=True,
        ))
    else:
        callbacks.append(BestWeights())

    model = fc.build(cfg, d["X_tr"].shape[1])
    t0 = time.perf_counter()
    history = model.fit(
        x_fit, y_fit,
        validation_data=(d["X_va"], d["y_va"]),
        epochs=cfg["max_epochs"],
        batch_size=cfg["batch"],
        class_weight=cw,
        callbacks=callbacks,
        verbose=0,
    ).history
    seconds = round(time.perf_counter() - t0, 1)

    keras_curve = [float(v) for v in history["val_pr_auc"]]
    best_epoch = int(np.argmax(keras_curve)) + 1
    p_va = model.predict(d["X_va"], verbose=0).ravel()
    th = fc._threshold_metrics(d["y_va"], p_va)
    val_pr = fc._ap(d["y_va"], p_va)
    res = dict(
        seed=seed,
        val_pr=val_pr,
        recall=th["recall"],
        precision=th["precision"],
        f1=th["f1"],
        best_epoch=best_epoch,
        n_params=int(model.count_params()),
        seconds=seconds,
    )
    return model, p_va, res, cw, d


def cache_report_val_pr(cfg, seed):
    hit = fc.cached_rows(cfg, seed)
    if hit is None:
        return None
    return float(hit.iloc[0]["report_val_pr"])


def write_card(path, card):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(py(card), f, ensure_ascii=False, indent=2)


def main():
    _, cfg, line = step09_compute()
    cfg = fc.cfg_of(**cfg)
    model_name = MODEL_NAME
    fc.header("10단계 · 최종 모델 동결", fc.cfg_line(cfg))
    print(f"학습률 원값: {cfg['lr']!r}")
    print(line)
    print(f"동결 시드: {DEPLOY_SEED}")
    print(f"저장 위치: {OUT_DIR}")
    print("flow_runs.csv 에는 쓰지 않음. test_final.csv 는 읽지 않음.")

    raw = load_raw()
    d_preview = fc.load_data()
    print(f"[data] train {raw['n_tr']}행, validation {raw['n_va']}행, "
          f"입력 {len(d_preview['columns'])}차원")

    cache_pr = cache_report_val_pr(cfg, DEPLOY_SEED)
    if cache_pr is None:
        print("[중단] flow_runs.csv에서 같은 설정·시드 42 행을 찾지 못함")
        return 1
    print(f"[캐시] seed {DEPLOY_SEED} report_val_pr {cache_pr:.6f}")

    model, p, res, cw, d = train_keep_model(cfg, DEPLOY_SEED)
    print(f"  seed {DEPLOY_SEED}: val PR-AUC {res['val_pr']:.6f} "
          f"(캐시 {cache_pr:.6f}) best_epoch {res['best_epoch']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, "mlp_final.keras")
    card_path = os.path.join(OUT_DIR, "model_card.json")
    model.save(model_path)
    joblib.dump(d["scaler"], os.path.join(OUT_DIR, "scaler.joblib"))
    with open(os.path.join(OUT_DIR, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"columns": d["columns"], "categorical": CATEGORICAL,
             "categories": raw["categories"]},
            f, ensure_ascii=False, indent=2,
        )
    pd.DataFrame({
        "row_id": np.arange(len(p)),
        "split": "validation",
        "y_true": d["y_va"],
        "y_probability": p,
        "model_name": model_name,
    }).to_csv(os.path.join(OUT_DIR, "validation_predictions_mlp.csv"), index=False)

    tf, keras, _ = fc.tf_modules()
    loaded = keras.models.load_model(model_path)
    p_reload = loaded.predict(d["X_va"], verbose=0).ravel()
    reload_pr = fc._ap(d["y_va"], p_reload)

    if C1_DIR not in sys.path:
        sys.path.insert(0, C1_DIR)
    import predict_mlp
    session_ok = False
    session_p = None
    session_err = None
    try:
        art = predict_mlp.load_artifacts(OUT_DIR)
        row = raw["raw_va"].drop(columns=["Revenue"]).iloc[0].to_dict()
        session_p = float(predict_mlp.predict_session(row, art))
        session_ok = np.isfinite(session_p)
    except Exception as e:
        session_err = str(e)

    check_cache = same6(res["val_pr"], cache_pr)
    check_reload = same6(res["val_pr"], reload_pr)
    check_session = bool(session_ok)
    print("[확인] 이번 학습 검증 PR-AUC vs flow_runs.csv seed 42 report_val_pr (소수 6자리)")
    print(f"  이번 {res['val_pr']:.6f}  캐시 {cache_pr:.6f}  "
          f"{'일치' if check_cache else '불일치'}")
    print("[확인] 저장 모델 재예측 검증 PR-AUC")
    print(f"  학습 {res['val_pr']:.6f}  재적재 {reload_pr:.6f}  "
          f"{'일치' if check_reload else '불일치'}")
    print("[확인] predict_mlp 세션 1건")
    if session_err:
        print(f"  오류: {session_err}  불일치")
    else:
        print(f"  확률 {session_p:.6f}  {'일치' if check_session else '불일치'}")

    all_ok = check_cache and check_reload and check_session
    print("일치" if all_ok else "불일치")

    config = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    card = dict(
        model_name=model_name,
        model_version=MODEL_VERSION,
        frozen_at=dt.date.today().isoformat(),
        created_script="step10_freeze.py",
        deploy_seed=DEPLOY_SEED,
        config=config,
        config_note=(
            f"{fc.cfg_line(cfg)}, ReLU+he_normal, 출력 sigmoid, "
            "이진 교차 엔트로피, 최대 에폭 후 검증 PR-AUC 최고 에폭 가중치"
        ),
        class_weight=cw,
        n_params=res["n_params"],
        input_dim=len(d["columns"]),
        seed=DEPLOY_SEED,
        val_pr=res["val_pr"],
        recall=res["recall"],
        precision=res["precision"],
        f1=res["f1"],
        best_epoch=res["best_epoch"],
        data=dict(
            train="data/final/train_clean.csv", train_rows=raw["n_tr"],
            validation="data/final/validation_final.csv", validation_rows=raw["n_va"],
            test="data/final/test_final.csv (이 스크립트는 읽지 않음)",
        ),
        sources=["members/C1/flow/flow_runs.csv", "members/C1/flow/step09_epoch.py"],
        notes=[
            "y_probability는 클래스 가중치로 학습해 실제 구매 확률보다 높게 나온다. 순위 판단용 점수로 해석할 것.",
            "임계값 0.5는 기본값일 뿐이며, 최종 임계값은 B가 검증 세트로 정한다(테스트 결과로 고르지 말 것).",
            "테스트 세트는 동결 후 1회만 평가한다.",
            "시연 입력은 predict_mlp.predict_session()을 쓸 것 (세션 1건 원핫 문제 처리됨). "
            "모델 폴더는 members/C1/final_model 이다.",
        ],
        versions=dict(
            tensorflow=tf.__version__, keras=keras.__version__,
            numpy=np.__version__, pandas=pd.__version__,
        ),
        reproduce_checks=dict(
            cache_pr_auc=dict(
                ok=check_cache, this=res["val_pr"], cache=cache_pr,
            ),
            reload_pr_auc=dict(
                ok=check_reload, this=res["val_pr"], reload=reload_pr,
            ),
            predict_session=dict(
                ok=check_session, probability=session_p, error=session_err,
            ),
        ),
    )
    write_card(card_path, card)

    print(f"\n저장 위치: {OUT_DIR}")
    for fn in sorted(os.listdir(OUT_DIR)):
        fp = os.path.join(OUT_DIR, fn)
        print(f"  - {fn}  {os.path.getsize(fp)} bytes")

    if not all_ok:
        print("[중단] 재현 확인 불일치. model_card.json에 표시함.")
        return 1
    print(f"\n[완료] 동결 모델(seed {DEPLOY_SEED}) 검증 PR-AUC {res['val_pr']:.6f}, "
          f"최고 에폭 {res['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
