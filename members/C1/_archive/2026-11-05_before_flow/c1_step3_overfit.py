"""
3단계 과적합 대응 (새 흐름).

출발: Adam 0.01, 배치 128, 최대 30에폭, 드롭아웃 0, 클래스 가중치, 시드 42/43/44.
최고 에폭은 Keras val_pr_auc, 보고 점수는 sklearn AP (c1_flow_common).

사용법:
  python members/C1/c1_step3_overfit.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

C1_DIR = os.path.dirname(os.path.abspath(__file__))
if C1_DIR not in sys.path:
    sys.path.insert(0, C1_DIR)

import c1_flow_common as flow
from c1_ver2_adam import tf, keras

START = flow.default_cfg(
    hidden=(28, 56, 28), dropout=0.0, drop_where="none",
    optimizer="Adam", lr=0.01, batch=128, max_epochs=30,
    early_stop=0, class_weight="balanced",
)
CANDIDATES = [
    ("드롭아웃 0 (출발)", dict(START)),
    ("드롭아웃 0.2", flow.default_cfg(**dict(START, dropout=0.2, drop_where="all"))),
    ("드롭아웃 0.3", flow.default_cfg(**dict(START, dropout=0.3, drop_where="all"))),
    ("조기종료 p5", flow.default_cfg(**dict(START, early_stop=5, max_epochs=100))),
    ("조기종료 p10", flow.default_cfg(**dict(START, early_stop=10, max_epochs=100))),
]
OFFICIAL_MEAN = 0.6907


def official_adam01():
    runs = pd.read_csv(os.path.join(C1_DIR, "ver2_adam_runs.csv"))

    def ok(cfg):
        return (
            cfg.get("optimizer") == "Adam"
            and abs(float(cfg.get("lr")) - 0.01) < 1e-12
            and cfg.get("batch") == 128
            and cfg.get("max_epochs") == 30
            and float(cfg.get("dropout", 0)) == 0.0
            and cfg.get("early_stop") == 0
            and cfg.get("imbalance") == "class_weight"
            and cfg.get("hidden") == [28, 56, 28]
        )

    hit = runs[runs["cfg"].map(lambda s: ok(json.loads(s)))].drop_duplicates("seed")
    return hit.sort_values("seed")


def check_repro(start_df):
    print("\n[재현 확인] Adam 0.01, 배치 128, 30에폭, 드롭아웃 0, 클래스 가중치")
    official = official_adam01()
    reports = flow.seed_reports(start_df)
    print(f"{'seed':>6} {'이번 val_pr':>14} {'csv val_pr':>14} {'차이':>12} "
          f"{'이번 best':>10} {'csv best':>9}")
    for seed in flow.SEEDS:
        got = reports.loc[reports["seed"] == seed].iloc[0]
        ref = official.loc[official["seed"] == seed].iloc[0]
        d_pr = float(got["val_pr"]) - float(ref["val_pr"])
        same_pr = round(float(got["val_pr"]), 6) == round(float(ref["val_pr"]), 6)
        same_ep = int(got["best_epoch"]) == int(ref["best_epoch"])
        print(f"{seed:6d} {float(got['val_pr']):14.6f} {float(ref['val_pr']):14.6f} {d_pr:12.6f}  "
              f"{int(got['best_epoch']):10d} {int(ref['best_epoch']):9d}  "
              f"{'일치' if same_pr and same_ep else '불일치'}")
    mean = float(reports["val_pr"].mean())
    std = float(reports["val_pr"].std(ddof=1))
    print(f"이번 요약 {mean:.4f}±{std:.4f}  기대 평균 {OFFICIAL_MEAN:.4f}  "
          f"{'일치' if round(mean, 4) == OFFICIAL_MEAN else '불일치'}")

    print("\n[에폭별] ver2_adam_runs.csv val_pr_curve(Keras) vs val_pr_keras")
    for seed in flow.SEEDS:
        ours = start_df[start_df["seed"] == seed].sort_values("epoch")
        curve = json.loads(official.loc[official["seed"] == seed, "val_pr_curve"].iloc[0])
        n = min(len(ours), len(curve))
        a = ours["val_pr_keras"].to_numpy(dtype=float)[:n]
        b = np.array(curve[:n], dtype=float)
        diff = np.abs(a - b)
        i = int(np.argmax(diff)) if n else 0
        print(f"  seed {seed}: 최대 |차이| {diff.max() if n else float('nan'):.6g}  "
              f"(에폭 {i + 1}, keras {a[i]:.6f} csv {b[i]:.6f})")

    cw_path = os.path.join(C1_DIR, "figures", "opt_lr_cw_curve.csv")
    print("\n[에폭별] opt_lr_cw_curve.csv val_pr vs val_pr_sk")
    cw = pd.read_csv(cw_path)
    sub = cw[(cw["optimizer"] == "Adam") & np.isclose(cw["lr"], 0.01)]
    for seed in flow.SEEDS:
        ours = start_df[start_df["seed"] == seed].sort_values("epoch")
        ref = sub[sub["seed"] == seed].sort_values("epoch")
        n = min(len(ours), len(ref))
        a = ours["val_pr_sk"].to_numpy(dtype=float)[:n]
        b = ref["val_pr"].to_numpy(dtype=float)[:n]
        diff = np.abs(a - b)
        i = int(np.argmax(diff)) if n else 0
        print(f"  seed {seed}: 최대 |차이| {diff.max() if n else float('nan'):.6g}  "
              f"(에폭 {i + 1}, sk {a[i]:.6f} csv {b[i]:.6f})")


def print_symptoms(start_df):
    print("\n=== 3 과적합 증상 (출발 조건) ===")
    for seed, sub in start_df.groupby("seed"):
        sub = sub.sort_values("epoch")
        r = sub.iloc[0]
        gap = float(r["report_train_pr"]) - float(r["report_val_pr"])
        i_min = int(sub["val_loss"].idxmin())
        min_ep = int(sub.loc[i_min, "epoch"])
        min_loss = float(sub.loc[i_min, "val_loss"])
        at30 = sub[sub["epoch"] == 30]
        loss30 = float(at30["val_loss"].iloc[0]) if len(at30) else float(sub["val_loss"].iloc[-1])
        rise = (loss30 - min_loss) / min_loss * 100.0
        print(f"  seed {int(seed)}: Keras 최고 {int(r['best_epoch'])}에폭  "
              f"학습PR−검증PR {gap:+.3f}  "
              f"검증손실 최저 {min_ep}에폭  30에폭 증가율 {rise:+.1f}%")


def main():
    t0 = time.perf_counter()
    d = flow.load_data()
    cw = flow.class_weight_of(d["y_tr"], "balanced")
    print(f"[env] tensorflow {tf.__version__}  keras {keras.__version__}")
    print(flow.key_of(START))
    print(f"class_weight  미구매(0)={cw[0]:.4f}  구매(1)={cw[1]:.4f}")
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} (test 미사용)")

    print("\n[학습] 출발 조건", flush=True)
    start_df = flow.run_seeds(START, d)
    check_repro(start_df)
    print_symptoms(start_df)

    stats = []
    table = []
    for label, cfg in CANDIDATES:
        print(f"\n[학습] {label}", flush=True)
        df = start_df if label.startswith("드롭아웃 0 (출발)") else flow.run_seeds(cfg, d)
        reports = flow.seed_reports(df)
        cells, val_mean, val_std = flow.combo_cells(label, reports)
        table.append(cells)
        stats.append({"label": label, "cells": cells, "val_mean": val_mean, "val_std": val_std})

    print("\n=== 3 과적합 후보 ===")
    print("(28,56,28), Adam 0.01, 배치 128, 클래스 가중치")
    text, widths = flow.render_table(table)
    print(text)

    start_s = stats[0]
    top = max(stats, key=lambda s: s["val_mean"])
    diff = top["val_mean"] - start_s["val_mean"]
    tol = max(top["val_std"], start_s["val_std"])
    print(f"  -> 1위: {top['label']}")
    print(flow.pad(top["cells"][0], widths[0]) + "  " +
          "  ".join(flow.pad(top["cells"][i], widths[i], "right") for i in range(1, len(top["cells"]))))
    print(f"     출발 조건 차이 {diff:+.4f}, 두 표준편차 중 큰 값 {tol:.4f}")
    print(f"\n저장: {flow.RUNS_CSV}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
