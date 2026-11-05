"""
0단계 출발 설정에서 학습률만 바꾸는 그리드 서치 (교재 4.10.1).

학습과 에폭별 기록은 c1_initial_curve.train_one을 그대로 쓴다.
옵티마이저는 c1_ver2_adam.make_optimizer의 SGD(momentum=0.9, nesterov=True).

사용법:
  python c1_lr_sgd.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import c1_initial_curve as curve
import c1_ver2_adam as v2

C1_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(C1_DIR, "figures")
OUT_CSV = os.path.join(FIG_DIR, "lr_sgd_curve.csv")
OUT_PNG = os.path.join(FIG_DIR, "lr_sgd_val.png")
LRS = [0.00005, 0.001, 0.01, 0.1]
LEGEND = ["lr=0.00005", "lr=0.001", "lr=0.01", "lr=0.1"]
BASE_LR = 0.00005
OFFICIAL = {42: 0.321157, 43: 0.167873, 44: 0.295385}
COLS = ["lr", "seed", "epoch", "train_pr", "val_pr", "val_recall", "val_pos_rate", "loss", "val_loss"]

# train_one은 예측에 NaN이 있으면 average_precision_score가 예외를 낸다.
# 학습 루프는 그대로 두고, 그 경우만 NaN으로 기록한다.
_ap = curve.average_precision_score


def _ap_or_nan(y_true, y_score):
    y_score = np.asarray(y_score, dtype=float)
    if not np.isfinite(y_score).all():
        return float("nan")
    return float(_ap(y_true, y_score))


curve.average_precision_score = _ap_or_nan


def fmt(x):
    if x is None or not np.isfinite(x):
        return "NaN" if x is None or np.isnan(x) else str(x)
    return f"{x:.6f}"


def matches_base(rows):
    df = pd.DataFrame(rows)
    last = df[df["epoch"] == 30]
    print("\n[lr=0.00005, 30에폭 검증 PR-AUC]")
    print(f"{'seed':>6} {'이번':>12} {'기준':>12} {'차이':>12}")
    ok = True
    for seed, expected in OFFICIAL.items():
        got = float(last.loc[last["seed"] == seed, "val_pr"].iloc[0])
        diff = got - expected
        if round(got, 6) != expected:
            ok = False
        print(f"{seed:6d} {got:12.6f} {expected:12.6f} {diff:12.6f}")
    return ok


def mark_nonfinite_class_metrics(rows):
    """점수가 유한하지 않으면 임계값 0.5 지표도 NaN으로 둔다."""
    for row in rows:
        if not np.isfinite(row["val_pr"]):
            row["val_recall"] = float("nan")
            row["val_pos_rate"] = float("nan")


def train_lr(lr, d):
    cfg = dict(v2.INITIAL)
    cfg["lr"] = lr
    cfg["max_epochs"] = 30
    rows = []
    for seed in v2.SEEDS:
        t0 = time.perf_counter()
        print(f"\n[학습] lr={lr:g} seed {seed}", flush=True)
        part = curve.train_one(cfg, seed, d)
        mark_nonfinite_class_metrics(part)
        for row in part:
            row["lr"] = lr
        rows.extend(part)
        last = part[-1]
        print(f"  완료 ({time.perf_counter() - t0:.1f}초)  "
              f"30에폭 검증 PR-AUC {fmt(last['val_pr'])}  학습 PR-AUC {fmt(last['train_pr'])}",
              flush=True)
    return rows


def best_of(sub):
    finite = sub[np.isfinite(sub["val_pr"])]
    if finite.empty:
        return float("nan"), float("nan")
    i = finite["val_pr"].idxmax()
    return float(finite.loc[i, "val_pr"]), float(finite.loc[i, "epoch"])


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1))


def report(df):
    print("\n[학습률별 요약] 평균±표본표준편차")
    for lr in LRS:
        sub = df[np.isclose(df["lr"], lr)]
        print(f"\nlr={lr:g}")
        print(f"{'seed':>6} {'best_pr':>12} {'best_ep':>8} {'val_pr':>12} {'train_pr':>12} "
              f"{'recall':>10} {'pos_rate':>10} {'loss':>12} {'val_loss':>12}")
        best_prs, best_eps = [], []
        metrics = {k: [] for k in ["val_pr", "train_pr", "val_recall", "val_pos_rate", "loss", "val_loss"]}
        for seed in v2.SEEDS:
            one = sub[sub["seed"] == seed]
            best_pr, best_ep = best_of(one)
            last = one[one["epoch"] == 30].iloc[0]
            best_prs.append(best_pr)
            best_eps.append(best_ep)
            for key in metrics:
                metrics[key].append(float(last[key]))
            print(f"{seed:6d} {fmt(best_pr):>12} {fmt(best_ep):>8} {fmt(last['val_pr']):>12} "
                  f"{fmt(last['train_pr']):>12} {fmt(last['val_recall']):>10} {fmt(last['val_pos_rate']):>10} "
                  f"{fmt(last['loss']):>12} {fmt(last['val_loss']):>12}")
        b_m, b_s = mean_std(best_prs)
        e_m, e_s = mean_std(best_eps)
        print(f"{'평균':>6} {fmt(b_m):>12} {fmt(e_m):>8}")
        print(f"{'표준편차':>6} {fmt(b_s):>12} {fmt(e_s):>8}", end="")
        for key in metrics:
            m, s = mean_std(metrics[key])
            print(f"  {key} {fmt(m)}±{fmt(s)}", end="")
        print()


def plot(df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10.6, 5.6), dpi=200)
    for lr in LRS:
        sub = df[np.isclose(df["lr"], lr)]
        g = sub.groupby("epoch")["val_pr"].mean(skipna=False)
        plt.plot(g.index.to_numpy(), g.to_numpy())
    plt.title("Validation PR-AUC by learning rate (SGD+NAG)")
    plt.ylabel("PR-AUC")
    plt.xlabel("Epoch")
    plt.legend(LEGEND)
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()


def main():
    t0 = time.perf_counter()
    d = v2.load_data()
    print(v2.describe(dict(v2.INITIAL, lr=BASE_LR)))
    print("SGD는 공식 make_optimizer: momentum=0.9, nesterov=True")
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} (test 미사용)")
    print("학습률:", ", ".join(f"{lr:g}" for lr in LRS))

    rows = train_lr(BASE_LR, d)
    if not matches_base(rows):
        print("\n[중단] lr=0.00005가 0단계 30에폭 검증 PR-AUC와 다릅니다. 다른 학습률은 돌리지 않습니다.")
        print(f"전체 {time.perf_counter() - t0:.1f}초")
        return 1

    for lr in LRS[1:]:
        rows.extend(train_lr(lr, d))

    df = pd.DataFrame(rows)[COLS]
    os.makedirs(FIG_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot(df)
    report(df)
    print(f"\n저장: {OUT_CSV}")
    print(f"      {OUT_PNG}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
