"""
C2 담당 - MLP 정규화·불균형 대응 실험 (조기종료 / 드롭아웃 / 클래스 가중치)
기준 설정은 C1 확정값(2층x64, Adam 1e-3, 배치 32, 최대 50에폭, 조기종료 patience 5,
class_weight balanced, 원핫 A 26차원, 공식 분할)이고, 한 번에 한 요소만 바꾼다.
데이터 로딩·전처리는 c1_structure_experiment.load_data()를 그대로 재사용한다.
보류(test) 분할은 읽지 않는다. 주 지표는 PR-AUC, Recall/F1은 임계값 0.5 기준 보조 지표.

사용법:
  python c2_regularization_experiment.py --smoke   # 기준 조건 seed=42 한 번 + C1 결과와 재현성 비교
  python c2_regularization_experiment.py           # 전체 12조건 x 3시드 = 36회 (이어달리기 가능)

C1 폴더 위치: 이 파일과 같은 폴더, 또는 ../C1. 다르면 환경변수 C1_DIR로 지정.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import recall_score, f1_score, average_precision_score, precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("C1_DIR"), BASE_DIR, os.path.join(BASE_DIR, "..", "C1")):
    if cand and os.path.exists(os.path.join(cand, "c1_structure_experiment.py")):
        sys.path.insert(0, os.path.abspath(cand))
        break

import c1_structure_experiment as se  # noqa: E402

# ---- 기준 설정 (C1 확정, C2 동의) ----
N_LAYERS = 2
UNITS = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 50
ES_MONITOR = "val_pr_auc"
ES_MODE = "max"
BASE_PATIENCE = 5
SEED_GRID = [42, 43, 44]

CSV_PATH = os.path.join(BASE_DIR, "c2_regularization_results.csv")
CSV_COLUMNS = [
    "group", "condition", "es_patience", "dropout", "drop_where", "class_weight", "seed",
    "recall", "precision", "f1", "pr_auc",
    "epochs", "best_epoch", "hit_max_epoch", "train_seconds", "n_params", "input_dim",
]


def cond(name, group, es=BASE_PATIENCE, dropout=0.0, where="none", cw="balanced"):
    """es=None이면 조기종료 없음. where는 'all'(모든 은닉층 뒤) 또는 'last'(마지막 은닉층 뒤)."""
    return {"condition": name, "group": group, "es_patience": es,
            "dropout": dropout, "drop_where": where, "class_weight": cw}


CONDITIONS = [
    cond("base", "기준"),
    # 조기종료
    cond("es_none", "조기종료", es=None),
    cond("es_p3", "조기종료", es=3),
    cond("es_p10", "조기종료", es=10),
    # 드롭아웃 (0.0은 기준과 동일하므로 따로 돌리지 않는다)
    cond("do_all_0.2", "드롭아웃", dropout=0.2, where="all"),
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
    for i in range(N_LAYERS):
        seq.append(layers.Dense(UNITS, activation="relu"))
        is_last = i == N_LAYERS - 1
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

    model = build_model(c, data["input_dim"])
    callbacks = []
    if c["es_patience"] is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=ES_MONITOR, mode=ES_MODE,
                patience=c["es_patience"], restore_best_weights=True,
            )
        )

    t0 = time.perf_counter()
    history = model.fit(
        data["X_train_s"], data["y_train"],
        validation_data=(data["X_val_s"], data["y_val"]),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
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
        "group": c["group"], "condition": c["condition"],
        "es_patience": "none" if c["es_patience"] is None else c["es_patience"],
        "dropout": c["dropout"], "drop_where": c["drop_where"],
        "class_weight": c["class_weight"], "seed": seed,
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "epochs": n_epochs, "best_epoch": best_epoch,
        "hit_max_epoch": bool(n_epochs == MAX_EPOCHS),
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


def compare_smoke_to_c1(row):
    """기준 조건은 C1의 2층x64 seed=42 결과와 같아야 한다 (드롭아웃 층을 아예 넣지 않으므로)."""
    ref = pd.read_csv(se.CSV_PATH)
    ref = ref[(ref["layers"] == 2) & (ref["units"] == 64) & (ref["seed"] == 42)]
    if ref.empty:
        print("[재현성] structure_results.csv에 2층x64 seed=42 행이 없음")
        return False
    ref = ref.iloc[0]
    ok = True
    print("[재현성] C2 base seed=42 vs C1 구조 실험 2x64 seed=42")
    for field in ["recall", "f1", "pr_auc", "epochs"]:
        a, b = row[field], ref[field]
        same = int(a) == int(b) if field == "epochs" else np.isclose(float(a), float(b), rtol=1e-12, atol=1e-12)
        ok &= bool(same)
        print(f"  {field}: c2={a} c1={b} -> {'일치' if same else '불일치'}")
    return ok


def fmt(mean, std):
    return f"{mean:.4f}±NA" if pd.isna(std) else f"{mean:.4f}±{std:.4f}"


def print_summary():
    df = pd.read_csv(CSV_PATH)
    order = {c["condition"]: i for i, c in enumerate(CONDITIONS)}
    g = df.groupby("condition").agg(
        group=("group", "first"), n=("seed", "count"),
        pr_m=("pr_auc", "mean"), pr_s=("pr_auc", "std"),
        rc_m=("recall", "mean"), rc_s=("recall", "std"),
        f1_m=("f1", "mean"), f1_s=("f1", "std"),
        ep=("epochs", "mean"), best=("best_epoch", "mean"), hit=("hit_max_epoch", "sum"),
    )
    g = g.loc[sorted(g.index, key=lambda k: order[k])]
    base_pr = g.loc["base", "pr_m"] if "base" in g.index else np.nan

    print("\n[조건별 평균±표준편차] (표본 표준편차 n-1, 실험 순서대로)")
    print(f"{'condition':>12} {'group':>10} {'n':>2} {'PR-AUC':>16} {'Δ기준':>8} "
          f"{'Recall':>16} {'F1':>16} {'epochs':>7} {'best':>6} {'hit_max':>7}")
    for name, r in g.iterrows():
        print(f"{name:>12} {r['group']:>10} {int(r['n']):>2} {fmt(r['pr_m'], r['pr_s']):>16} "
              f"{r['pr_m'] - base_pr:>+8.4f} {fmt(r['rc_m'], r['rc_s']):>16} "
              f"{fmt(r['f1_m'], r['f1_s']):>16} {r['ep']:>7.1f} {r['best']:>6.1f} {int(r['hit']):>7}")
    print("\n※ Δ기준이 PR-AUC 표준편차(대략 0.002~0.009)와 비슷하면 우열을 단정하지 말 것.")


def main(smoke=False):
    data = se.load_data()
    print(f"[data] input_dim={data['input_dim']} class_weight(balanced)={data['class_weight']}")
    combos = all_combos()
    done = load_done_keys()
    target = [(CONDITIONS[0], 42)] if smoke else combos

    last_row = None
    for c, seed in target:
        k = combos.index((c, seed)) + 1
        if not smoke and (c["condition"], seed) in done:
            print(f"[{k}/{len(combos)}] {c['condition']} seed={seed} SKIP (CSV에 있음)")
            continue
        row = run_one(c, seed, data)
        last_row = row
        if not smoke:
            append_csv(row)
        print(f"[{k}/{len(combos)}] {c['condition']} seed={seed} epochs={row['epochs']} "
              f"best={row['best_epoch']} hit_max={row['hit_max_epoch']} sec={row['train_seconds']:.1f} "
              f"Recall={row['recall']:.4f} Prec={row['precision']:.4f} "
              f"F1={row['f1']:.4f} PR-AUC={row['pr_auc']:.4f}")

    if smoke:
        if not compare_smoke_to_c1(last_row):
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
                        help="base seed=42 한 번 실행 후 C1 결과와 비교")
    main(smoke=parser.parse_args().smoke)
