"""
옵티마이저 × 학습률 그리드 + 클래스 가중치.

c1_opt_lr.py와 같은 12조합·시드·평가 방식이고, class_weight만 켠다.
기존 스크립트와 기존 결과 파일은 읽기만 한다. 36회 모두 새로 학습한다.

사용법:
  python members/C1/c1_opt_lr_cw.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

C1_DIR = os.path.dirname(os.path.abspath(__file__))
if C1_DIR not in sys.path:
    sys.path.insert(0, C1_DIR)

import c1_lr_sgd_summary as summary
import c1_ver2_adam as v2
from c1_initial_curve import EpochSnapshot
from c1_ver2_adam import keras, tf

FIG_DIR = os.path.join(C1_DIR, "figures")
OUT_CSV = os.path.join(FIG_DIR, "opt_lr_cw_curve.csv")

LRS = [0.00005, 0.001, 0.01, 0.1]
LR_TEXT = {0.00005: "0.00005", 0.001: "0.001", 0.01: "0.01", 0.1: "0.1"}
LR_LABELS = ["lr=0.00005", "lr=0.001", "lr=0.01", "lr=0.1"]
OPTIMIZERS = ["SGD", "RMSprop", "Adam"]
OPT_LABEL = {"SGD": "SGD+NAG", "RMSprop": "RMSprop", "Adam": "Adam"}
COLS = ["optimizer", "lr", "seed", "epoch", "train_pr", "val_pr",
        "val_recall", "val_pos_rate", "loss", "val_loss"]
CHECK_COMBOS = [
    ("SGD", 0.00005),
    ("SGD", 0.001),
    ("SGD", 0.01),
    ("RMSprop", 0.001),
    ("RMSprop", 0.01),
    ("Adam", 0.001),
    ("Adam", 0.01),
]
OFFICIAL_SGD_TINY = {42: 0.402303, 43: 0.259990, 44: 0.367295}


def class_weight_of(y):
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


def ap_or_nan(y_true, y_score):
    y_score = np.asarray(y_score, dtype=float)
    if not np.isfinite(y_score).all():
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def mark_nonfinite_class_metrics(rows):
    for row in rows:
        if not np.isfinite(row["val_pr"]):
            row["val_recall"] = float("nan")
            row["val_pos_rate"] = float("nan")


def fmt(x):
    if x is None or not np.isfinite(x):
        return "NaN"
    return f"{x:.6f}"


def train_one(cfg, seed, d, cw):
    """c1_initial_curve.train_one과 같고 class_weight만 켠다."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    snap = EpochSnapshot()
    model = v2.build(cfg, d["X_tr"].shape[1])
    model.fit(
        d["X_tr"], d["y_tr"],
        validation_data=(d["X_va"], d["y_va"]),
        epochs=cfg["max_epochs"],
        batch_size=cfg["batch"],
        class_weight=cw,
        callbacks=[v2.BestWeights(), snap],
        verbose=0,
    )
    rows = []
    for epoch, (weights, logs) in enumerate(zip(snap.weights, snap.logs), start=1):
        model.set_weights(weights)
        p_tr = model.predict(d["X_tr"], verbose=0).ravel()
        p_va = model.predict(d["X_va"], verbose=0).ravel()
        pred = (p_va >= 0.5).astype(int)
        rows.append(dict(
            seed=seed,
            epoch=epoch,
            train_pr=ap_or_nan(d["y_tr"], p_tr),
            val_pr=ap_or_nan(d["y_va"], p_va),
            val_recall=float(recall_score(d["y_va"], pred, zero_division=0)),
            val_pos_rate=float(pred.mean()),
            loss=float(logs["loss"]),
            val_loss=float(logs["val_loss"]),
        ))
    mark_nonfinite_class_metrics(rows)
    return rows


def train_seeds(optimizer, lr, seeds, d, cw):
    cfg = dict(v2.INITIAL)
    cfg["optimizer"] = optimizer
    cfg["lr"] = lr
    cfg["imbalance"] = "class_weight"
    cfg["max_epochs"] = 30
    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        print(f"\n[학습] {optimizer} lr={LR_TEXT[lr]} seed {seed}", flush=True)
        part = train_one(cfg, seed, d, cw)
        for row in part:
            row["optimizer"] = optimizer
            row["lr"] = lr
        rows.extend(part)
        last = part[-1]
        print(f"  완료 ({time.perf_counter() - t0:.1f}초)  "
              f"30에폭 검증 PR-AUC {fmt(last['val_pr'])}  "
              f"학습 PR-AUC {fmt(last['train_pr'])}  "
              f"loss {fmt(last['loss'])}",
              flush=True)
    return rows


def collect(d, cw):
    rows = []
    for optimizer in OPTIMIZERS:
        for lr in LRS:
            rows.extend(train_seeds(optimizer, lr, v2.SEEDS, d, cw))
    df = pd.DataFrame(rows)[COLS]
    opt_rank = {name: i for i, name in enumerate(OPTIMIZERS)}
    df["_opt"] = df["optimizer"].map(opt_rank)
    df["_lr"] = df["lr"].map(lambda x: next(i for i, lr in enumerate(LRS) if np.isclose(x, lr)))
    df = df.sort_values(["_opt", "_lr", "seed", "epoch"], kind="mergesort")
    return df.drop(columns=["_opt", "_lr"]).reset_index(drop=True)


def row_at_best(sub):
    finite = sub[np.isfinite(sub["val_pr"])]
    if finite.empty:
        return sub.iloc[-1]
    return finite.loc[finite["val_pr"].idxmax()]


def combo_stats(df, optimizer, lr, label):
    sub = df[(df["optimizer"] == optimizer) & np.isclose(df["lr"], lr)]
    best = [row_at_best(one) for _, one in sub.groupby("seed")]
    val = np.array([float(r.val_pr) for r in best], dtype=float)
    train = np.array([float(r.train_pr) for r in best], dtype=float)
    recall = np.array([float(r.val_recall) for r in best], dtype=float)
    pos = np.array([float(r.val_pos_rate) for r in best], dtype=float)
    epoch = np.array([float(r.epoch) for r in best], dtype=float)
    val_mean = float(np.nanmean(val)) if np.isfinite(val).any() else float("nan")
    train_mean = float(np.nanmean(train)) if np.isfinite(train).any() else float("nan")
    val_std = float(np.nanstd(val, ddof=1)) if np.isfinite(val).sum() > 1 else float("nan")
    rec_mean = float(np.nanmean(recall)) if np.isfinite(recall).any() else float("nan")
    pos_mean = float(np.nanmean(pos)) if np.isfinite(pos).any() else float("nan")
    ep_mean = float(np.nanmean(epoch)) if np.isfinite(epoch).any() else float("nan")
    prauc = "NaN" if not np.isfinite(val_mean) else f"{val_mean:.4f}±{val_std:.4f}"
    cells = [
        label,
        prauc,
        "NaN" if not np.isfinite(train_mean) else f"{train_mean:.3f}",
        "NaN" if not (np.isfinite(train_mean) and np.isfinite(val_mean)) else f"{train_mean - val_mean:+.3f}",
        "NaN" if not np.isfinite(rec_mean) else f"{rec_mean:.3f}",
        "NaN" if not np.isfinite(pos_mean) else f"{pos_mean:.3f}",
        "NaN" if not np.isfinite(ep_mean) else f"{int(round(ep_mean))}/30",
    ]
    return {"optimizer": optimizer, "lr": lr, "val_mean": val_mean, "cells": cells}


def winners(stats):
    picked = []
    for optimizer in OPTIMIZERS:
        group = [s for s in stats if s["optimizer"] == optimizer]

        def score(item):
            return item["val_mean"] if np.isfinite(item["val_mean"]) else -np.inf

        picked.append(max(group, key=score))
    return picked


def column_widths(table):
    return [
        max(summary.disp_width(summary.HEADERS[i]), *(summary.disp_width(row[i]) for row in table))
        for i in range(len(summary.HEADERS))
    ]


def format_row(cells, widths):
    return "  ".join(
        summary.pad(cells[i], widths[i], summary.ALIGNS[i]) for i in range(len(cells))
    )


def print_sgd(stats):
    sgd = [s for s in stats if s["optimizer"] == "SGD"]
    table = [s["cells"] for s in sgd]
    print("=== 학습률 그리드 (SGD+NAG, 그 외 0단계 그대로) ===")
    print("(28,56,28), SGD+NAG, 배치 128, 최대 30에폭, 조기종료 없음, 클래스 가중치")
    print(summary.render(table))


def print_grid(stats):
    table = [s["cells"] for s in stats]
    widths = column_widths(table)
    print("=== 옵티마이저 × 학습률 그리드 (그 외 0단계 그대로) ===")
    print("(28,56,28), 배치 128, 최대 30에폭, 조기종료 없음, 클래스 가중치")
    print(summary.render(table))
    print("옵티마이저별 1위 학습률")
    for item in winners(stats):
        print(format_row(item["cells"], widths))


def cfg_ok(cfg, optimizer, lr):
    return (
        cfg.get("optimizer") == optimizer
        and abs(float(cfg.get("lr")) - lr) < 1e-12
        and cfg.get("hidden") == [28, 56, 28]
        and cfg.get("imbalance") == "class_weight"
        and cfg.get("batch") == 128
        and float(cfg.get("dropout", 0)) == 0.0
        and cfg.get("max_epochs") == 30
        and cfg.get("early_stop") == 0
    )


def load_official(optimizer, lr):
    runs = pd.read_csv(v2.RUNS_CSV)
    parsed = runs["cfg"].map(json.loads)
    hit = runs[parsed.map(lambda c: cfg_ok(c, optimizer, lr))].drop_duplicates("seed")
    return hit.sort_values("seed")


def check_official(df):
    print("\n[재현 확인] ver2_adam_runs.csv 클래스 가중치, batch 128, dropout 0, max_epochs 30")
    for optimizer, lr in CHECK_COMBOS:
        official = load_official(optimizer, lr)
        print(f"\n{optimizer} lr={LR_TEXT[lr]}")
        print(f"{'seed':>6} {'이번(공식best에폭)':>20} {'csv':>12} {'차이':>12}")
        if official.empty:
            print("  ver2_adam_runs.csv에서 같은 조건 행을 찾지 못함")
            continue
        for seed in v2.SEEDS:
            row = official[official["seed"] == seed]
            if row.empty:
                print(f"{seed:6d}  csv 시드 없음")
                continue
            expected = float(row["val_pr"].iloc[0])
            best_ep = int(row["best_epoch"].iloc[0])
            got_at = df[
                (df["optimizer"] == optimizer)
                & np.isclose(df["lr"], lr)
                & (df["seed"] == seed)
                & (df["epoch"] == best_ep)
            ]
            got = float(got_at["val_pr"].iloc[0]) if len(got_at) else float("nan")
            diff = got - expected
            same = np.isfinite(got) and round(got, 6) == round(expected, 6)
            flag = "일치" if same else "불일치"
            print(f"{seed:6d} {got:20.6f} {expected:12.6f} {diff:12.6f}  {flag}")
            if optimizer == "SGD" and abs(lr - 0.00005) < 1e-12:
                listed = OFFICIAL_SGD_TINY[seed]
                if round(got, 6) != listed:
                    print(f"         지정값 {listed:.6f} 과 차이 {got - listed:+.6f}")


def main():
    t0 = time.perf_counter()
    d = v2.load_data()
    cw = class_weight_of(d["y_tr"])
    cfg = dict(v2.INITIAL, imbalance="class_weight", max_epochs=30)
    print(f"[env] tensorflow {tf.__version__}  keras {keras.__version__}")
    print(v2.describe(cfg))
    print("옵티마이저는 c1_ver2_adam.make_optimizer")
    print(f"class_weight  미구매(0)={cw[0]:.4f}  구매(1)={cw[1]:.4f}  (train_clean balanced)")
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} (test 미사용)")
    print("12조합 × 시드 3 = 36회 모두 새로 학습")

    df = collect(d, cw)
    os.makedirs(FIG_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    check_official(df)

    sgd_stats = [
        combo_stats(df, "SGD", lr, LR_LABELS[i]) for i, lr in enumerate(LRS)
    ]
    grid_stats = [
        combo_stats(df, optimizer, lr, f"{OPT_LABEL[optimizer]} {LR_TEXT[lr]}")
        for optimizer in OPTIMIZERS for lr in LRS
    ]
    print()
    print_sgd(sgd_stats)
    print()
    print_grid(grid_stats)
    print(f"\n저장: {OUT_CSV}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
