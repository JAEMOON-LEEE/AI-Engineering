"""
출발점과 같고 클래스 가중치만 켠 학습 곡선.

c1_initial_curve.py와 같은 모델·시드·결정론 순서이고, fit()에 class_weight만 넣는다.
기존 스크립트와 기존 결과 파일은 읽기만 한다.

사용법:
  python members/C1/c1_cw_curve.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

C1_DIR = os.path.dirname(os.path.abspath(__file__))
if C1_DIR not in sys.path:
    sys.path.insert(0, C1_DIR)

import c1_ver2_adam as v2
from c1_initial_curve import EpochSnapshot
from c1_ver2_adam import keras, tf

FIG_DIR = os.path.join(C1_DIR, "figures")
OUT_CSV = os.path.join(FIG_DIR, "cw_curve.csv")
OUT30 = os.path.join(FIG_DIR, "cw_epoch30.png")
OUT300 = os.path.join(FIG_DIR, "cw_epoch300.png")
SEEDS = list(v2.SEEDS)
MAX_EPOCHS = 300
OFFICIAL30 = {42: 0.402303, 43: 0.259990, 44: 0.367295}


def class_weight_of(y):
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


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
    for epoch, weights in enumerate(snap.weights, start=1):
        model.set_weights(weights)
        p_tr = model.predict(d["X_tr"], verbose=0).ravel()
        p_va = model.predict(d["X_va"], verbose=0).ravel()
        rows.append(dict(
            seed=seed,
            epoch=epoch,
            train_pr=float(average_precision_score(d["y_tr"], p_tr)),
            val_pr=float(average_precision_score(d["y_va"], p_va)),
        ))
    return rows


def load_official_cw():
    runs = pd.read_csv(v2.RUNS_CSV)
    key = v2.key_of(dict(v2.INITIAL, imbalance="class_weight"))
    hit = runs[runs["cfg"] == key].drop_duplicates("seed")
    return hit, key


def check_epoch30(df):
    print("\n[확인] 30에폭 검증 PR-AUC vs ver2_adam_runs.csv 클래스 가중치 첫 실행")
    print(f"{'seed':>6} {'이번':>12} {'공식':>12} {'차이':>12}")
    last = df[df["epoch"] == 30]
    ok = True
    for seed in SEEDS:
        got = float(last.loc[last["seed"] == seed, "val_pr"].iloc[0])
        expected = OFFICIAL30[seed]
        diff = got - expected
        same = round(got, 6) == expected
        ok = ok and same
        flag = "일치" if same else "불일치"
        print(f"{seed:6d} {got:12.6f} {expected:12.6f} {diff:12.6f}  {flag}")
    ref, key = load_official_cw()
    if len(ref):
        print(f"공식 cfg key: {key}")
        for seed in SEEDS:
            row = ref[ref["seed"] == seed]
            if len(row):
                official = float(row["val_pr"].iloc[0])
                got = float(last.loc[last["seed"] == seed, "val_pr"].iloc[0])
                if round(got, 6) != round(official, 6):
                    print(f"  csv 원값 시드 {seed}: 이번 {got:.10f}  csv {official:.10f}  차이 {got - official:+.10f}")
    return ok


def summarize(df, last_epoch):
    sub = df[df["epoch"] <= last_epoch]
    end = sub[sub["epoch"] == last_epoch]
    mean = float(end["val_pr"].mean())
    std = float(end["val_pr"].std(ddof=1))
    print(f"\n=== {last_epoch}에폭 ===")
    print(f"검증 PR-AUC {mean:.4f}±{std:.4f}  (시드 평균, 표본 표준편차)")
    print("시드별 최고 검증 PR-AUC 에폭")
    for seed, one in sub.groupby("seed"):
        i = one["val_pr"].idxmax()
        row = one.loc[i]
        print(f"  seed {int(seed)}: {int(row['epoch'])}에폭  {row['val_pr']:.6f}")


def plot_slide(df, last_epoch, title, path):
    """c1_slide_figures._textbook_plot과 같은 스타일. Train 파랑, Validation 주황."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    g = df[df["epoch"] <= last_epoch].groupby("epoch")
    epochs = g.size().index.to_numpy()
    train_mean = g["train_pr"].mean().to_numpy()
    val_mean = g["val_pr"].mean().to_numpy()
    plt.figure(figsize=(10.6, 5.6), dpi=200)
    plt.plot(epochs, train_mean)
    plt.plot(epochs, val_mean)
    plt.title(title)
    plt.ylabel("PR-AUC")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    t0 = time.perf_counter()
    cfg = dict(v2.INITIAL)
    cfg["imbalance"] = "class_weight"
    cfg["max_epochs"] = MAX_EPOCHS
    d = v2.load_data()
    cw = class_weight_of(d["y_tr"])
    print(f"[env] tensorflow {tf.__version__}  keras {keras.__version__}")
    print(v2.describe(cfg))
    print(f"class_weight  미구매(0)={cw[0]:.4f}  구매(1)={cw[1]:.4f}  (train_clean balanced)")
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} 입력 {d['X_tr'].shape[1]}차원 (test 미사용)")

    rows = []
    for seed in SEEDS:
        t_seed = time.perf_counter()
        print(f"\n[학습] seed {seed}, {MAX_EPOCHS}에폭", flush=True)
        part = train_one(cfg, seed, d, cw)
        rows.extend(part)
        at30 = next(r for r in part if r["epoch"] == 30)
        last = part[-1]
        print(f"  완료 ({time.perf_counter() - t_seed:.1f}초)  "
              f"30에폭 검증 {at30['val_pr']:.6f}  "
              f"300에폭 검증 {last['val_pr']:.6f}", flush=True)

    df = pd.DataFrame(rows)
    check_epoch30(df)
    summarize(df, 30)
    summarize(df, 300)

    os.makedirs(FIG_DIR, exist_ok=True)
    df[["seed", "epoch", "train_pr", "val_pr"]].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot_slide(df, 30, "Model PR-AUC (30 epochs)", OUT30)
    plot_slide(df, 300, "Model PR-AUC (300 epochs)", OUT300)
    print(f"\n저장: {OUT_CSV}")
    print(f"      {OUT30}")
    print(f"      {OUT300}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
