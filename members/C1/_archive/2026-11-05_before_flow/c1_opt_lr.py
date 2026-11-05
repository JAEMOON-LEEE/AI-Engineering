"""
옵티마이저 × 학습률 그리드 (교재 4.10.1, 5.4).

학습과 에폭별 기록은 c1_initial_curve.train_one을 그대로 쓴다.
옵티마이저는 c1_ver2_adam.make_optimizer를 build()가 호출한다.
  SGD     = SGD(learning_rate=lr, momentum=0.9, nesterov=True)
  RMSprop = RMSprop(learning_rate=lr)
  Adam    = Adam(learning_rate=lr)

SGD 12조합은 lr_sgd_curve.csv를 그대로 붙인다.
본 실험 전에 SGD lr=0.01 시드 42만 다시 돌려 30에폭 검증 PR-AUC를 대조한다.

사용법:
  python c1_opt_lr.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import c1_initial_curve as curve
import c1_lr_sgd as sgd
import c1_lr_sgd_summary as summary
import c1_ver2_adam as v2

C1_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(C1_DIR, "figures")
SGD_CSV = os.path.join(FIG_DIR, "lr_sgd_curve.csv")
OUT_CSV = os.path.join(FIG_DIR, "opt_lr_curve.csv")
OUT_PNG = os.path.join(FIG_DIR, "opt_lr_best.png")

LRS = [0.00005, 0.001, 0.01, 0.1]
LR_TEXT = {0.00005: "0.00005", 0.001: "0.001", 0.01: "0.01", 0.1: "0.1"}
OPTIMIZERS = ["SGD", "RMSprop", "Adam"]
OPT_LABEL = {"SGD": "SGD+NAG", "RMSprop": "RMSprop", "Adam": "Adam"}
NEW_OPTIMIZERS = ["RMSprop", "Adam"]
COLS = ["optimizer", "lr", "seed", "epoch", "train_pr", "val_pr",
        "val_recall", "val_pos_rate", "loss", "val_loss"]
CHECK_LR = 0.01
CHECK_SEED = 42


def candidate(optimizer, lr):
    return f"{OPT_LABEL[optimizer]} {LR_TEXT[lr]}"


def train_seeds(optimizer, lr, seeds, d):
    cfg = dict(v2.INITIAL)
    cfg["optimizer"] = optimizer
    cfg["lr"] = lr
    cfg["max_epochs"] = 30
    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        print(f"\n[학습] {optimizer} lr={LR_TEXT[lr]} seed {seed}", flush=True)
        part = curve.train_one(cfg, seed, d)
        sgd.mark_nonfinite_class_metrics(part)
        for row in part:
            row["optimizer"] = optimizer
            row["lr"] = lr
        rows.extend(part)
        last = part[-1]
        print(f"  완료 ({time.perf_counter() - t0:.1f}초)  "
              f"30에폭 검증 PR-AUC {sgd.fmt(last['val_pr'])}  "
              f"학습 PR-AUC {sgd.fmt(last['train_pr'])}  "
              f"loss {sgd.fmt(last['loss'])}",
              flush=True)
    return rows


def confirm_sgd(d):
    """SGD lr=0.01 시드 42의 30에폭 검증 PR-AUC가 기존 CSV와 같은지."""
    saved = pd.read_csv(SGD_CSV)
    hit = saved[np.isclose(saved["lr"], CHECK_LR) & (saved["seed"] == CHECK_SEED) & (saved["epoch"] == 30)]
    expected = float(hit["val_pr"].iloc[0])
    print(f"[확인] SGD lr={LR_TEXT[CHECK_LR]} seed {CHECK_SEED}를 한 번 학습합니다.", flush=True)
    rows = train_seeds("SGD", CHECK_LR, [CHECK_SEED], d)
    got = float([r["val_pr"] for r in rows if r["epoch"] == 30][0])
    same = round(got, 6) == round(expected, 6)
    print(f"[확인] 30에폭 검증 PR-AUC  이번 {got:.10f}  csv {expected:.10f}  "
          f"차이 {got - expected:.3e}")
    print("[확인] 일치" if same else "[확인] 불일치 — 본 실험은 진행하지 않습니다.")
    return same


def load_sgd_rows():
    saved = pd.read_csv(SGD_CSV)
    saved.insert(0, "optimizer", "SGD")
    return saved[COLS]


def collect(d):
    frames = [load_sgd_rows()]
    fresh = []
    for optimizer in NEW_OPTIMIZERS:
        for lr in LRS:
            fresh.extend(train_seeds(optimizer, lr, v2.SEEDS, d))
    if fresh:
        frames.append(pd.DataFrame(fresh)[COLS])
    df = pd.concat(frames, ignore_index=True)
    opt_rank = {name: i for i, name in enumerate(OPTIMIZERS)}
    df["_opt"] = df["optimizer"].map(opt_rank)
    df["_lr"] = df["lr"].map(lambda x: next(i for i, lr in enumerate(LRS) if np.isclose(x, lr)))
    df = df.sort_values(["_opt", "_lr", "seed", "epoch"], kind="mergesort")
    return df.drop(columns=["_opt", "_lr"]).reset_index(drop=True)


def combo_stats(df, optimizer, lr):
    sub = df[(df["optimizer"] == optimizer) & np.isclose(df["lr"], lr)]
    best = [summary.row_at_best(one) for _, one in sub.groupby("seed")]
    val = np.array([float(r.val_pr) for r in best])
    train = np.array([float(r.train_pr) for r in best])
    recall = np.array([float(r.val_recall) for r in best])
    pos = np.array([float(r.val_pos_rate) for r in best])
    epoch = np.array([float(r.epoch) for r in best])
    val_mean = float(val.mean())
    train_mean = float(train.mean())
    cells = [
        candidate(optimizer, lr),
        f"{val_mean:.4f}±{float(val.std(ddof=1)):.4f}",
        f"{train_mean:.3f}",
        f"{train_mean - val_mean:+.3f}",
        f"{float(recall.mean()):.3f}",
        f"{float(pos.mean()):.3f}",
        f"{int(round(float(epoch.mean())))}/30",
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


def print_summary(stats):
    table = [s["cells"] for s in stats]
    widths = column_widths(table)
    print("=== 옵티마이저 × 학습률 그리드 (그 외 0단계 그대로) ===")
    print("(28,56,28), 배치 128, 최대 30에폭, 조기종료 없음, 불균형 처리 없음")
    print(summary.render(table))
    print("옵티마이저별 1위 학습률")
    for item in winners(stats):
        print(format_row(item["cells"], widths))


def plot(stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10.6, 5.6), dpi=200)
    xs = np.array(LRS, dtype=float)
    for optimizer in OPTIMIZERS:
        ys = [next(s["val_mean"] for s in stats if s["optimizer"] == optimizer and s["lr"] == lr) for lr in LRS]
        plt.plot(xs, ys, marker="o")
    plt.xscale("log")
    plt.title("Best validation PR-AUC by optimizer and learning rate")
    plt.xlabel("Learning rate")
    plt.ylabel("PR-AUC")
    plt.legend(["SGD+NAG", "RMSprop", "Adam"])
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()


def main():
    t0 = time.perf_counter()
    d = v2.load_data()
    print(v2.describe(dict(v2.INITIAL)))
    print("옵티마이저는 c1_ver2_adam.make_optimizer")
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} (test 미사용)")
    if not confirm_sgd(d):
        print(f"전체 {time.perf_counter() - t0:.1f}초")
        return 1

    df = collect(d)
    os.makedirs(FIG_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    stats = [combo_stats(df, optimizer, lr) for optimizer in OPTIMIZERS for lr in LRS]
    plot(stats)
    print(f"\n저장: {OUT_CSV}")
    print(f"      {OUT_PNG}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    print()
    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
