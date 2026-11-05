"""
3단계 출발점(Adam 0.01)의 에폭별 손실 곡선.

조건은 c1_step2_opt_lr.py의 Adam 0.01과 같다.
학습·에폭 기록은 c1_flow_common을 쓰고, c1_overfit_history.csv가 있으면 재사용한다.

사용법:
  python members/C1/c1_step3_overfit_curve.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

C1_DIR = os.path.dirname(os.path.abspath(__file__))
if C1_DIR not in sys.path:
    sys.path.insert(0, C1_DIR)

import c1_slide_figures as slides

FIG_DIR = os.path.join(C1_DIR, "figures")
HISTORY_CSV = os.path.join(C1_DIR, "c1_overfit_history.csv")
OUT_PNG = os.path.join(FIG_DIR, "overfit_loss_curve.png")
SEEDS = (42, 43, 44)
EXPECTED = {42: 0.688512, 43: 0.701328, 44: 0.682238}
HIST_COLS = ["seed", "epoch", "loss", "val_loss", "train_pr", "val_pr", "best_epoch"]
TRAIN_COLOR = "#D9731F"
VAL_COLOR = "#2F5BD8"
TEXT = "#31343A"
MUTED = "#6A6E75"


def start_cfg():
    import c1_flow_common as flow

    return flow.default_cfg(
        hidden=(28, 56, 28),
        dropout=0.0,
        drop_where="none",
        optimizer="Adam",
        lr=0.01,
        batch=128,
        max_epochs=30,
        early_stop=0,
        class_weight="balanced",
    )


def history_complete(df):
    if df is None or df.empty or any(col not in df.columns for col in HIST_COLS):
        return False
    for seed in SEEDS:
        epochs = set(df.loc[df["seed"] == seed, "epoch"].astype(int))
        if epochs != set(range(1, 31)):
            return False
    return True


def load_history():
    if not os.path.isfile(HISTORY_CSV):
        return None
    df = pd.read_csv(HISTORY_CSV)
    if not history_complete(df):
        return None
    print(f"[재사용] {HISTORY_CSV}")
    return df.sort_values(["seed", "epoch"]).reset_index(drop=True)


def history_from_flow():
    import c1_flow_common as flow

    print("[기록] c1_flow_runs.csv에 있으면 학습하지 않습니다.", flush=True)
    ran = flow.run_seeds(start_cfg(), flow.load_data())
    rows = ran.rename(columns={"val_pr_sk": "val_pr"})[HIST_COLS]
    rows = rows.sort_values(["seed", "epoch"]).reset_index(drop=True)
    rows.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
    print(f"저장: {HISTORY_CSV}")
    return rows


def best_val_pr(df, seed):
    one = df[df["seed"] == seed]
    epoch = int(one["best_epoch"].iloc[0])
    value = float(one.loc[one["epoch"] == epoch, "val_pr"].iloc[0])
    return epoch, value


def check(df):
    print("[확인] 시드별 최고 에폭 검증 PR-AUC  vs  c1_flow_runs.csv Adam 0.01")
    print(f"{'seed':>6} {'이번':>12} {'기준':>12}  결과")
    ok = True
    for seed in SEEDS:
        _epoch, value = best_val_pr(df, seed)
        same = round(value, 6) == EXPECTED[seed]
        ok = ok and same
        print(f"{seed:6d} {value:12.6f} {EXPECTED[seed]:12.6f}  {'일치' if same else '불일치'}")
    return ok


def seed_summary(df):
    rows = []
    for seed in SEEDS:
        one = df[df["seed"] == seed].sort_values("epoch")
        i = one["val_loss"].idxmin()
        lowest = one.loc[i]
        last = float(one.loc[one["epoch"] == 30, "val_loss"].iloc[0])
        base = float(lowest["val_loss"])
        rise = (last - base) / base * 100.0
        rows.append(dict(
            seed=seed,
            min_epoch=int(lowest["epoch"]),
            min_loss=base,
            loss_30=last,
            rise=rise,
            best_epoch=int(one["best_epoch"].iloc[0]),
        ))
    return rows


def print_summary(rows):
    print("[검증 손실] 최저 에폭, 30에폭, 증가율, 최고 PR-AUC 에폭")
    print(f"{'seed':>6} {'최저에폭':>8} {'최저손실':>10} {'30에폭손실':>12} {'증가율':>8} {'최고PR에폭':>10}")
    for row in rows:
        print(f"{row['seed']:6d} {row['min_epoch']:8d} {row['min_loss']:10.4f} "
              f"{row['loss_30']:12.4f} {row['rise']:+7.1f}% {row['best_epoch']:10d}")


def _face(font_paths, weight, size):
    from matplotlib import font_manager

    prop = font_manager.FontProperties(fname=font_paths[weight])
    prop.set_size(size)
    return prop


def plot(df, rows):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    font_paths = slides._gothic_paths()
    for path in font_paths.values():
        font_manager.fontManager.addfont(path)
    label_face = _face(font_paths, "bold", 15)
    tick_face = _face(font_paths, "bold", 12)
    note_face = _face(font_paths, "extrabold", 13)

    epochs = np.arange(1, 31)
    train = np.vstack([
        df.loc[df["seed"] == seed].sort_values("epoch")["loss"].to_numpy(dtype=float)
        for seed in SEEDS
    ])
    val = np.vstack([
        df.loc[df["seed"] == seed].sort_values("epoch")["val_loss"].to_numpy(dtype=float)
        for seed in SEEDS
    ])
    train_mean, val_mean = train.mean(axis=0), val.mean(axis=0)
    min_i = int(np.argmin(val_mean))
    min_epoch = int(epochs[min_i])
    rise = (val_mean[-1] - val_mean[min_i]) / val_mean[min_i] * 100.0
    pr_epoch = float(np.mean([row["best_epoch"] for row in rows]))

    fig = plt.figure(figsize=(10.6, 5.6), dpi=200)
    fig.patch.set_facecolor("#F5F6F7")
    ax = fig.add_axes([0.12, 0.14, 0.82, 0.78])
    ax.set_facecolor("#F5F6F7")
    ax.fill_between(epochs, train.min(axis=0), train.max(axis=0), color=TRAIN_COLOR, alpha=0.13, linewidth=0)
    ax.fill_between(epochs, val.min(axis=0), val.max(axis=0), color=VAL_COLOR, alpha=0.13, linewidth=0)
    ax.plot(epochs, train_mean, color=TRAIN_COLOR, linewidth=2.4, label="학습 손실")
    ax.plot(epochs, val_mean, color=VAL_COLOR, linewidth=2.4, label="검증 손실")
    ax.scatter([min_epoch], [val_mean[min_i]], s=46, color=VAL_COLOR, zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.axvline(pr_epoch, color=MUTED, linestyle=":", linewidth=1.4, alpha=0.9, zorder=1)
    ax.text(min_epoch + 0.45, val_mean[min_i] + 0.012, f"최저 {min_epoch}에폭",
            fontproperties=note_face, color=TEXT, va="bottom", ha="left")
    ax.text(29.55, val_mean[-1] + 0.012, f"+{rise:.0f}%",
            fontproperties=note_face, color=VAL_COLOR, va="bottom", ha="right")
    ax.text(pr_epoch + 0.35, 0.97, "PR-AUC 최고", transform=ax.get_xaxis_transform(),
            fontproperties=note_face, color=MUTED, va="top", ha="left")

    ax.set_xlim(1, 30)
    ax.set_xlabel("에폭", fontproperties=label_face, color="#4A4E55")
    ax.set_ylabel("손실", fontproperties=label_face, color="#4A4E55", labelpad=10)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(tick_face)
        tick.set_color(TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E3E5EA")
    ax.spines["bottom"].set_color("#E3E5EA")
    ax.tick_params(color="#E3E5EA", length=0)
    legend = ax.legend(loc="upper right", frameon=False, prop=_face(font_paths, "bold", 13))
    for text in legend.get_texts():
        text.set_color(TEXT)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"저장: {OUT_PNG}")


def main():
    df = load_history()
    if df is None:
        df = history_from_flow()
    matched = check(df)
    rows = seed_summary(df)
    print_summary(rows)
    if not matched:
        print("[중단] 최고 에폭 검증 PR-AUC가 c1_flow_runs.csv와 다릅니다. 그림은 저장하지 않습니다.")
        return 1
    plot(df, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
