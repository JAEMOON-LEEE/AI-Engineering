"""
C2 담당 - 최종 학습 조건 위에서 MLP 정규화·불균형 대응 실험 재실행
(드롭아웃 / 조기종료 / 클래스 가중치)

기준 설정은 C1이 확정한 최종 학습 조건이다.
  은닉층 (28, 56, 28), he_normal 초기화, Adam lr=0.01/3, 배치 64, 최대 30에폭, 시드 42/43/44.
기본 평가 방식은 "조기종료 없이 30에폭까지 학습한 뒤, 검증 PR-AUC가 가장 높았던 에폭의
가중치로 되돌려 평가"(eval=best)이다. 한 번에 한 요소만 바꾼다(12조건 x 3시드 = 36회).
데이터 로딩·전처리는 c1_structure_experiment.load_data()를 그대로 재사용한다.
보류(test) 분할은 읽지 않는다. 주 지표는 PR-AUC, Recall/Precision/F1은 임계값 0.5 기준 보조 지표.

eval 방식
  best : BestWeights 콜백, epochs=30 (val_pr_auc 최고 에폭 가중치로 복원)
  last : 콜백 없음, epochs=30 (마지막 에폭 가중치 그대로)
  es5  : EarlyStopping(val_pr_auc, max, patience=5, restore_best_weights), epochs=100
  es10 : EarlyStopping(val_pr_auc, max, patience=10, restore_best_weights), epochs=100

사용법:
  python c2_final_experiment.py --smoke   # base를 시드 42/43/44로 돌려 C1 ver2_adam_runs.csv와 비교
  python c2_final_experiment.py           # 전체 12조건 x 3시드 (이어달리기 가능)

C1 폴더 위치: ../C1 (또는 같은 폴더). 다르면 환경변수 C1_DIR로 지정.
결과 CSV: c2_final_results.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import recall_score, f1_score, average_precision_score, precision_score

tf.config.experimental.enable_op_determinism()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
C1_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "C1"))
for cand in (os.environ.get("C1_DIR"), BASE_DIR, os.path.join(BASE_DIR, "..", "C1")):
    if cand and os.path.exists(os.path.join(cand, "c1_structure_experiment.py")):
        C1_DIR = os.path.abspath(cand)
        sys.path.insert(0, C1_DIR)
        break

import c1_structure_experiment as se  # noqa: E402

# ---- 기준 설정 (C1 최종 학습 조건) ----
HIDDEN = (28, 56, 28)
LEARNING_RATE = 0.01 / 3
BATCH_SIZE = 64
MAX_EPOCHS = 30
ES_MAX_EPOCHS = 100          # es5 / es10 전용 최대 에폭
ES_MONITOR = "val_pr_auc"
ES_MODE = "max"
SEED_GRID = [42, 43, 44]

# eval 방식별 설정: (최대 에폭, 조기종료 patience)
EVAL_MODES = {
    "best": {"epochs": MAX_EPOCHS, "patience": None},
    "last": {"epochs": MAX_EPOCHS, "patience": None},
    "es5": {"epochs": ES_MAX_EPOCHS, "patience": 5},
    "es10": {"epochs": ES_MAX_EPOCHS, "patience": 10},
}

CSV_PATH = os.path.join(BASE_DIR, "c2_final_results.csv")
CSV_COLUMNS = [
    "group", "condition", "eval", "es_patience", "dropout", "drop_where", "class_weight", "seed",
    "recall", "precision", "f1", "pr_auc",
    "epochs", "best_epoch", "hit_max_epoch", "train_seconds", "n_params", "input_dim",
]

# C1 ver2_adam_runs.csv에서 비교 대상 행을 고르는 조건 (cfg 열을 json.loads한 뒤 비교)
C1_RUNS_PATH = os.path.join(C1_DIR, "ver2_adam_runs.csv")
C1_CFG_MATCH = {"hidden": [28, 56, 28], "dropout": 0.2, "optimizer": "Adam"}
C1_SEED_COL = "seed"
C1_VALUE_COL = "val_pr"


class BestWeights(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.best, self.weights = -np.inf, None

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get("val_pr_auc")
        if v is not None and v > self.best:
            self.best, self.weights = v, self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.weights is not None:
            self.model.set_weights(self.weights)


def cond(name, group, eval_mode="best", dropout=0.2, where="all", cw="balanced"):
    """base에서 한 번에 하나만 바꾼다. where는 'all'(모든 은닉층 뒤) / 'last'(마지막 은닉층 뒤) / 'none'."""
    return {"condition": name, "group": group, "eval": eval_mode,
            "dropout": dropout, "drop_where": where, "class_weight": cw}


CONDITIONS = [
    cond("base", "기준"),
    # 평가 방식 / 조기종료
    cond("last_epoch", "평가방식", eval_mode="last"),
    cond("es_p5", "조기종료", eval_mode="es5"),
    cond("es_p10", "조기종료", eval_mode="es10"),
    # 드롭아웃
    cond("do_none", "드롭아웃", dropout=0.0, where="none"),
    cond("do_all_0.3", "드롭아웃", dropout=0.3, where="all"),
    cond("do_all_0.5", "드롭아웃", dropout=0.5, where="all"),
    cond("do_last_0.2", "드롭아웃", dropout=0.2, where="last"),
    cond("do_last_0.3", "드롭아웃", dropout=0.3, where="last"),
    cond("do_last_0.5", "드롭아웃", dropout=0.5, where="last"),
    # 클래스 가중치
    cond("cw_none", "클래스가중치", cw="none"),
    cond("cw_sqrt", "클래스가중치", cw="sqrt"),
]


def make_class_weight(mode, balanced):
    if mode == "none":
        return None
    if mode == "balanced":
        return balanced
    if mode == "sqrt":
        return {k: float(np.sqrt(v)) for k, v in balanced.items()}
    raise ValueError(f"지원하지 않는 class_weight: {mode}")


def build_model(c, input_dim):
    seq = [layers.Input(shape=(input_dim,))]
    for i, u in enumerate(HIDDEN):
        seq.append(layers.Dense(u, activation="relu", kernel_initializer="he_normal"))
        is_last = i == len(HIDDEN) - 1
        if c["dropout"] > 0 and (c["drop_where"] == "all" or (c["drop_where"] == "last" and is_last)):
            seq.append(layers.Dropout(c["dropout"]))
    seq.append(layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(seq)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def run_one(c, seed, data):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)

    mode = EVAL_MODES[c["eval"]]
    max_epochs = mode["epochs"]

    model = build_model(c, data["input_dim"])
    callbacks = []
    if c["eval"] == "best":
        callbacks.append(BestWeights())
    elif mode["patience"] is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=ES_MONITOR, mode=ES_MODE,
                patience=mode["patience"], restore_best_weights=True,
            )
        )

    t0 = time.perf_counter()
    history = model.fit(
        data["X_train_s"], data["y_train"],
        validation_data=(data["X_val_s"], data["y_val"]),
        epochs=max_epochs, batch_size=BATCH_SIZE,
        class_weight=make_class_weight(c["class_weight"], data["class_weight"]),
        callbacks=callbacks, verbose=0,
    )
    train_seconds = time.perf_counter() - t0

    n_epochs = len(history.history["loss"])
    best_epoch = int(np.argmax(history.history["val_pr_auc"])) + 1

    y_prob = model.predict(data["X_val_s"], verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = data["y_val"]

    return {
        "group": c["group"], "condition": c["condition"], "eval": c["eval"],
        "es_patience": "none" if mode["patience"] is None else mode["patience"],
        "dropout": c["dropout"], "drop_where": c["drop_where"],
        "class_weight": c["class_weight"], "seed": seed,
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "epochs": n_epochs, "best_epoch": best_epoch,
        "hit_max_epoch": bool(n_epochs == max_epochs),
        "train_seconds": float(train_seconds),
        "n_params": int(model.count_params()),
        "input_dim": int(data["input_dim"]),
    }


def all_combos():
    return [(c, seed) for c in CONDITIONS for seed in SEED_GRID]


def load_done_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return set()
    return set(zip(df["condition"].astype(str), df["seed"].astype(int)))


def append_csv(row):
    pd.DataFrame([{k: row[k] for k in CSV_COLUMNS}]).to_csv(
        CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False
    )


def _cfg_matches(cfg_text):
    try:
        cfg = json.loads(cfg_text)
    except (TypeError, ValueError):
        return False
    return all(k in cfg and cfg[k] == v for k, v in C1_CFG_MATCH.items())


def compare_smoke_to_c1(rows):
    """base를 시드별로 돌린 결과와 C1 ver2_adam_runs.csv의 시드별 val_pr을 비교한다."""
    if not os.path.exists(C1_RUNS_PATH):
        print(f"[재현성] C1 파일을 찾을 수 없음: {C1_RUNS_PATH}")
        return False
    ref = pd.read_csv(C1_RUNS_PATH)
    for col in ("cfg", C1_SEED_COL, C1_VALUE_COL):
        if col not in ref.columns:
            print(f"[재현성] ver2_adam_runs.csv에 '{col}' 열이 없음. 열 목록: {list(ref.columns)}")
            return False

    ref = ref[ref["cfg"].apply(_cfg_matches)]
    print(f"[재현성] C2 base vs C1 {os.path.basename(C1_RUNS_PATH)} "
          f"(조건 {C1_CFG_MATCH}, 일치 행 {len(ref)}개)")
    if ref.empty:
        print("[재현성] 조건에 맞는 C1 행이 없음")
        return False

    ok = True
    for row in rows:
        seed = row["seed"]
        cand = ref[ref[C1_SEED_COL].astype(int) == int(seed)]
        if cand.empty:
            print(f"  seed={seed}: C1에 해당 시드 행 없음 -> 불일치")
            ok = False
            continue
        if len(cand) > 1:
            print(f"  seed={seed}: C1 후보 행이 {len(cand)}개 (cfg가 더 구체적이어야 할 수 있음), 하나라도 맞으면 일치로 봄")
        a = row["pr_auc"]
        same = any(np.isclose(float(a), float(b), rtol=1e-12, atol=1e-12) for b in cand[C1_VALUE_COL])
        ok &= bool(same)
        print(f"  seed={seed}: c2 pr_auc={a!r} c1 val_pr={list(cand[C1_VALUE_COL])} -> "
              f"{'일치' if same else '불일치'}")
    return ok


def fmt(mean, std):
    return f"{mean:.4f}±NA" if pd.isna(std) else f"{mean:.4f}±{std:.4f}"


def print_summary():
    df = pd.read_csv(CSV_PATH)
    order = {c["condition"]: i for i, c in enumerate(CONDITIONS)}
    g = df.groupby("condition").agg(
        group=("group", "first"), ev=("eval", "first"), n=("seed", "count"),
        pr_m=("pr_auc", "mean"), pr_s=("pr_auc", "std"),
        rc_m=("recall", "mean"), rc_s=("recall", "std"),
        f1_m=("f1", "mean"), f1_s=("f1", "std"),
        ep=("epochs", "mean"), best=("best_epoch", "mean"), hit=("hit_max_epoch", "sum"),
    )
    g = g.loc[sorted(g.index, key=lambda k: order[k])]
    base_pr = g.loc["base", "pr_m"] if "base" in g.index else np.nan

    print("\n[조건별 평균±표준편차] (표본 표준편차 n-1, 실험 순서대로)")
    print(f"{'condition':>12} {'group':>10} {'eval':>5} {'n':>2} {'PR-AUC':>16} {'Δ기준':>8} "
          f"{'Recall':>16} {'F1':>16} {'epochs':>7} {'best':>6} {'hit_max':>7}")
    for name, r in g.iterrows():
        print(f"{name:>12} {r['group']:>10} {r['ev']:>5} {int(r['n']):>2} "
              f"{fmt(r['pr_m'], r['pr_s']):>16} {r['pr_m'] - base_pr:>+8.4f} "
              f"{fmt(r['rc_m'], r['rc_s']):>16} {fmt(r['f1_m'], r['f1_s']):>16} "
              f"{r['ep']:>7.1f} {r['best']:>6.1f} {int(r['hit']):>7}")
    print("\n※ Δ기준이 PR-AUC 표준편차와 비슷하면 우열을 단정하지 말 것.")


def main(smoke=False):
    data = se.load_data()
    print(f"[data] input_dim={data['input_dim']} class_weight(balanced)={data['class_weight']}")
    combos = all_combos()
    done = load_done_keys()
    base = CONDITIONS[0]
    target = [(base, s) for s in SEED_GRID] if smoke else combos

    smoke_rows = []
    for c, seed in target:
        k = combos.index((c, seed)) + 1
        if not smoke and (c["condition"], seed) in done:
            print(f"[{k}/{len(combos)}] {c['condition']} seed={seed} SKIP (CSV에 있음)")
            continue
        row = run_one(c, seed, data)
        if smoke:
            smoke_rows.append(row)
        else:
            append_csv(row)
        print(f"[{k}/{len(combos)}] {c['condition']} seed={seed} eval={row['eval']} "
              f"epochs={row['epochs']} best={row['best_epoch']} hit_max={row['hit_max_epoch']} "
              f"sec={row['train_seconds']:.1f} Recall={row['recall']:.4f} "
              f"Prec={row['precision']:.4f} F1={row['f1']:.4f} PR-AUC={row['pr_auc']:.4f}")

    if smoke:
        if not compare_smoke_to_c1(smoke_rows):
            print("[재현성] 불일치. 여기서 중단.")
            sys.exit(1)
        print("[재현성] 일치. 전체 실험 진행 가능.")
        return

    if len(load_done_keys()) == len(combos):
        print_summary()
    else:
        print(f"완료 {len(load_done_keys())}/{len(combos)}. 다시 실행하면 이어서 진행함.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="base를 시드 42/43/44로 실행 후 C1 ver2_adam_runs.csv와 비교")
    main(smoke=parser.parse_args().smoke)
