"""
C1 담당 - 후보 구조·학습률 × 드롭아웃 재검증
데이터는 c1_structure_experiment.load_data()를 그대로 사용한다. test는 읽지 않음.
기존 구조/옵티마이저/배치 스크립트는 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import openpyxl

C1_DIR = os.path.dirname(os.path.abspath(__file__))
if C1_DIR not in sys.path:
    sys.path.insert(0, C1_DIR)

import c1_structure_experiment as se

BATCH_SIZE = se.BATCH_SIZE
MAX_EPOCHS = se.MAX_EPOCHS
EARLY_STOPPING = se.EARLY_STOPPING
ES_MONITOR = se.ES_MONITOR
ES_MODE = se.ES_MODE
ES_PATIENCE = se.ES_PATIENCE
RESTORE_BEST_WEIGHTS = se.RESTORE_BEST_WEIGHTS

C1_002_PR_AUC = 0.661
DROPOUT_GRID = [0.0, 0.2, 0.3]
SEED_GRID = [42, 43, 44, 45, 46]
CANDIDATES = {
    "A": {"hidden": (64, 64), "optimizer": "Adam", "lr": 1e-3},
    "B": {"hidden": (32, 32), "optimizer": "Adam", "lr": 1e-3},
    "C": {"hidden": (32,), "optimizer": "Adam", "lr": 1e-3},
    "D": {"hidden": (64, 64), "optimizer": "Adam", "lr": 1e-2},
    "E": {"hidden": (128, 128), "optimizer": "Adam", "lr": 1e-3},
}

PREP_VERSION = "official_v1(data/final, Month·VisitorType 원핫, 26차원)"
CSV_COLUMNS = [
    "candidate", "hidden", "optimizer", "lr", "dropout", "seed",
    "n_params", "epochs", "best_epoch",
    "pr_auc", "f1", "recall", "precision",
    "hit_max_epoch", "train_seconds", "input_dim",
]
LOG_PATH = os.path.join(C1_DIR, "experiment_log_C1.xlsx")
CSV_PATH = os.path.join(C1_DIR, "revalidation_results.csv")
N_RUNS = len(CANDIDATES) * len(DROPOUT_GRID) * len(SEED_GRID)


def hidden_str(hidden):
    return "(" + ",".join(str(u) for u in hidden) + ")"


def build_model(hidden, dropout, lr, input_dim):
    """드롭아웃은 C2 do_all과 같이 모든 은닉 Dense 뒤에만 붙인다. dropout=0이면 Dropout 층을 넣지 않는다."""
    seq = [layers.Input(shape=(input_dim,))]
    for units in hidden:
        seq.append(layers.Dense(units, activation="relu"))
        if dropout > 0:
            seq.append(layers.Dropout(dropout))
    seq.append(layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(seq)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def run_one(candidate, dropout, seed, data):
    spec = CANDIDATES[candidate]
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)

    model = build_model(spec["hidden"], dropout, spec["lr"], data["input_dim"])
    callbacks = []
    if EARLY_STOPPING:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=ES_MONITOR,
                mode=ES_MODE,
                patience=ES_PATIENCE,
                restore_best_weights=RESTORE_BEST_WEIGHTS,
            )
        )

    t0 = time.perf_counter()
    history = model.fit(
        data["X_train_s"],
        data["y_train"],
        validation_data=(data["X_val_s"], data["y_val"]),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=data["class_weight"],
        callbacks=callbacks,
        verbose=0,
    )
    train_seconds = time.perf_counter() - t0

    n_epochs = len(history.history["loss"])
    best_epoch = int(np.argmax(history.history["val_pr_auc"])) + 1 if "val_pr_auc" in history.history else None
    y_prob = model.predict(data["X_val_s"], verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = data["y_val"]

    return {
        "candidate": candidate,
        "hidden": hidden_str(spec["hidden"]),
        "optimizer": spec["optimizer"],
        "lr": spec["lr"],
        "dropout": float(dropout),
        "seed": int(seed),
        "n_params": int(model.count_params()),
        "epochs": n_epochs,
        "best_epoch": best_epoch,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "hit_max_epoch": bool(n_epochs == MAX_EPOCHS),
        "train_seconds": float(train_seconds),
        "input_dim": int(data["input_dim"]),
    }


def all_combos():
    return [
        (cand, dropout, seed)
        for cand in CANDIDATES
        for dropout in DROPOUT_GRID
        for seed in SEED_GRID
    ]


def combo_key(candidate, dropout, seed):
    return (str(candidate), round(float(dropout), 10), int(seed))


def unique_results(df):
    """집계·엑셀은 (후보, dropout, seed)당 1행만 사용한다."""
    out = df.drop_duplicates(["candidate", "dropout", "seed"], keep="first").copy()
    cand_order = {c: i for i, c in enumerate(CANDIDATES)}
    out["_c"] = out["candidate"].map(cand_order)
    out = out.sort_values(["_c", "dropout", "seed"]).drop(columns="_c")
    return out.reset_index(drop=True)


def load_done_keys(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(combo_key(a, b, c) for a, b, c in zip(df["candidate"], df["dropout"], df["seed"]))


def append_csv(path, row):
    pd.DataFrame([{k: row[k] for k in CSV_COLUMNS}]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )


MEMO_RE = re.compile(r"재검증\s+([A-E]).*dropout=([0-9.]+)")


def excel_existing_keys(ws):
    keys = set()
    row = 2
    while ws.cell(row=row, column=1).value is not None:
        if ws.cell(row=row, column=3).value == "재검증":
            memo = str(ws.cell(row=row, column=18).value or "")
            seed = ws.cell(row=row, column=14).value
            m = MEMO_RE.search(memo)
            if m:
                try:
                    keys.add(combo_key(m.group(1), m.group(2), seed))
                except (TypeError, ValueError):
                    pass
        row += 1
    return keys, row


def write_excel_from_csv(csv_path, log_path):
    results = unique_results(pd.read_csv(csv_path))
    if len(results) < N_RUNS:
        print(f"고유 조합이 {len(results)}행이라 엑셀 {N_RUNS}행 추가를 건너뜀.")
        return

    early_stop_log = (
        f"{ES_MONITOR}, patience={ES_PATIENCE}, mode={ES_MODE}, restore_best={RESTORE_BEST_WEIGHTS}"
        if EARLY_STOPPING
        else "없음"
    )
    try:
        wb = openpyxl.load_workbook(log_path)
        ws = wb["실험로그"]
        existing, next_row = excel_existing_keys(ws)
        added = 0
        first_id = last_id = None
        for _, row in results.iterrows():
            key = combo_key(row["candidate"], row["dropout"], row["seed"])
            if key in existing:
                continue
            existing.add(key)
            hidden = tuple(int(x) for x in str(row["hidden"]).strip("()").split(",") if x)
            exp_id = f"C1-{next_row - 1:03d}"
            if first_id is None:
                first_id = exp_id
            last_id = exp_id
            values = [
                exp_id,
                date.today().isoformat(),
                "재검증",
                f"tensorflow(keras) {tf.__version__}",
                PREP_VERSION,
                int(row["input_dim"]),
                len(hidden),
                hidden[-1] if hidden else None,
                str(row["optimizer"]),
                float(row["lr"]),
                BATCH_SIZE,
                MAX_EPOCHS,
                early_stop_log,
                int(row["seed"]),
                round(float(row["recall"]), 4),
                round(float(row["f1"]), 4),
                round(float(row["pr_auc"]), 4),
                f"공식 분할 기준, 재검증 {row['candidate']}, dropout={float(row['dropout'])}",
            ]
            for col, value in enumerate(values, start=1):
                ws.cell(row=next_row, column=col, value=value)
            next_row += 1
            added += 1
        wb.save(log_path)
        if added:
            print(f"[엑셀] 실험로그에 {added}행 추가 {first_id}~{last_id}")
        else:
            print("[엑셀] 추가할 재검증 행 없음")
    except PermissionError:
        print("엑셀을 닫고 다시 실행하세요")
        sys.exit(1)


def print_summary(csv_path):
    df = unique_results(pd.read_csv(csv_path))
    grouped = (
        df.groupby(["candidate", "dropout"], as_index=False)
        .agg(
            n=("seed", "count"),
            pr_mean=("pr_auc", "mean"),
            pr_std=("pr_auc", "std"),
            f1_mean=("f1", "mean"),
            rec_mean=("recall", "mean"),
            prec_mean=("precision", "mean"),
            ep_mean=("epochs", "mean"),
            n_params=("n_params", "first"),
        )
        .sort_values("pr_mean", ascending=False)
    )
    print("\n[(후보, dropout)별 평균±표준편차 ddof=1, n=시드 수, PR-AUC 내림차순]")
    print(
        f"{'cand':>4} {'drop':>5} {'n':>2} {'PR-AUC':>16} "
        f"{'F1평균':>8} {'Recall평균':>10} {'Prec평균':>8} {'에폭평균':>8} {'params':>8}"
    )
    for _, r in grouped.iterrows():
        std = "NA" if pd.isna(r["pr_std"]) else f"{r['pr_std']:.4f}"
        print(
            f"{r['candidate']:>4} {float(r['dropout']):>5.1f} {int(r['n']):>2} "
            f"{r['pr_mean']:.4f}±{std:>6} "
            f"{r['f1_mean']:>8.4f} {r['rec_mean']:>10.4f} {r['prec_mean']:>8.4f} "
            f"{r['ep_mean']:>8.1f} {int(r['n_params']):>8}"
        )


def main(smoke=False):
    data = se.load_data()
    print(f"[data] input_dim={data['input_dim']} (test 미사용)")
    combos = all_combos()
    done = load_done_keys(CSV_PATH)
    target = [("A", 0.0, 42)] if smoke else combos

    last_row = None
    t_all0 = time.perf_counter()
    for candidate, dropout, seed in target:
        k = combos.index((candidate, dropout, seed)) + 1
        key = combo_key(candidate, dropout, seed)
        if not smoke and key in done:
            print(f"[{k}/{N_RUNS}] {candidate} drop={dropout} seed={seed} SKIP")
            continue
        row = run_one(candidate, dropout, seed, data)
        last_row = row
        if not smoke:
            append_csv(CSV_PATH, row)
        print(
            f"[{k}/{N_RUNS}] {row['candidate']} hidden={row['hidden']} "
            f"{row['optimizer']} lr={row['lr']} drop={row['dropout']} seed={row['seed']} "
            f"params={row['n_params']} epochs={row['epochs']} best={row['best_epoch']} "
            f"PR-AUC={row['pr_auc']:.4f} F1={row['f1']:.4f} "
            f"Recall={row['recall']:.4f} Prec={row['precision']:.4f} "
            f"sec={row['train_seconds']:.1f}"
        )

    if smoke:
        if last_row is None:
            print("[스모크] 결과를 얻지 못함")
            sys.exit(1)
        elapsed = time.perf_counter() - t_all0
        same = np.isclose(last_row["pr_auc"], C1_002_PR_AUC, atol=5e-4, rtol=0)
        print("\n[스모크] A / dropout=0 / seed=42")
        print(f"  val PR-AUC = {last_row['pr_auc']:.6f}")
        print(f"  C1-002 비교값 = {C1_002_PR_AUC} -> {'일치' if same else '불일치'}")
        print(f"  1회 소요(학습+평가) = {last_row['train_seconds']:.1f}초")
        print(f"  스크립트 전체(데이터 로드 포함) = {elapsed:.1f}초")
        print("  CSV/엑셀은 쓰지 않음. 전체 75회는 --smoke 없이 실행.")
        return

    done = load_done_keys(CSV_PATH)
    if len(done) == N_RUNS:
        write_excel_from_csv(CSV_PATH, LOG_PATH)
        print_summary(CSV_PATH)
    else:
        print(f"완료 {len(done)}/{N_RUNS}. 엑셀은 {N_RUNS}회 끝난 뒤에만 추가.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="A, dropout 0, seed 42만 실행")
    main(smoke=parser.parse_args().smoke)
