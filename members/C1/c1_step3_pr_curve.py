"""
c1_overfit_history.csv만 읽어 Adam 0.01 검증 PR-AUC 곡선을 그린다.
학습하지 않는다.

사용법:
  python members/C1/c1_step3_pr_curve.py
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
OUT_VAL = os.path.join(FIG_DIR, "overfit_val_pr_curve.png")
OUT_BOTH = os.path.join(FIG_DIR, "overfit_train_val_pr_curve.png")
SEEDS = (42, 43, 44)
# c1_step2_print.py 출력 3, Adam 0.01
EXPECTED_BEST = {42: 0.6885, 43: 0.7013, 44: 0.6822}
EXPECTED_E30_VAL = {42: 0.6662, 43: 0.6728, 44: 0.6445}
EXPECTED_E30_TRAIN = {42: 0.834, 43: 0.840, 44: 0.833}
EXPECTED_MARK = "12.3"
SEED_COLOR = {42: "#9BB6EF", 43: "#5B84E6", 44: "#274EAE"}
VAL_COLOR = "#2F5BD8"
TRAIN_COLOR = "#D9731F"
TEXT = "#31343A"
MUTED = "#6A6E75"
BG = "#F5F6F7"


def load_history():
    if not os.path.isfile(HISTORY_CSV):
        print(f"[중단] 기록 없음: {HISTORY_CSV}")
        return None
    df = pd.read_csv(HISTORY_CSV)
    needed = {"seed", "epoch", "train_pr", "val_pr", "best_epoch"}
    if needed - set(df.columns):
        print("[중단] c1_overfit_history.csv에 에폭별 PR 열이 없습니다.")
        return None
    for seed in SEEDS:
        epochs = set(df.loc[df["seed"] == seed, "epoch"].astype(int))
        if epochs != set(range(1, 31)):
            print(f"[중단] 시드 {seed}의 1~30에폭 기록이 없습니다.")
            return None
    return df.sort_values(["seed", "epoch"]).reset_index(drop=True)


def series(df, col):
    return np.vstack([
        df.loc[df["seed"] == seed].sort_values("epoch")[col].to_numpy(dtype=float)
        for seed in SEEDS
    ])


def seed_rows(df):
    rows = []
    for seed in SEEDS:
        one = df[df["seed"] == seed]
        best = int(one["best_epoch"].iloc[0])
        at = one.loc[one["epoch"] == best].iloc[0]
        last = one.loc[one["epoch"] == 30].iloc[0]
        rows.append(dict(
            seed=seed,
            best_epoch=best,
            best_val=float(at["val_pr"]),
            e30_val=float(last["val_pr"]),
            e30_train=float(last["train_pr"]),
        ))
    return rows


def mean_curve_best(val):
    return int(np.argmax(val.mean(axis=0))) + 1


def same(got, expected, digits):
    return round(got, digits) == expected


def print_table(rows, mark, curve_epoch):
    headers = ["시드", "최고에폭", "최고 검증PR", "30에폭 검증PR", "30에폭 학습PR"]
    body = [[
        str(row["seed"]),
        str(row["best_epoch"]),
        f"{row['best_val']:.4f}",
        f"{row['e30_val']:.4f}",
        f"{row['e30_train']:.3f}",
    ] for row in rows]
    widths = [max(len(headers[i]), *(len(line[i]) for line in body)) for i in range(5)]
    aligns = ["right"] * 5

    def cell(text, width, align):
        gap = width - len(text)
        return (" " * gap + text) if align == "right" else (text + " " * gap)

    print("  ".join(cell(headers[i], widths[i], aligns[i]) for i in range(5)))
    for line in body:
        print("  ".join(cell(line[i], widths[i], aligns[i]) for i in range(5)))
    print(f"시드별 최고 에폭 평균: {mark:.1f}")
    print(f"평균 곡선 기준 최고 에폭: {curve_epoch}")


def print_gaps(df):
    train = series(df, "train_pr").mean(axis=0)
    val = series(df, "val_pr").mean(axis=0)
    print("에폭  학습 평균  검증 평균    차이")
    for epoch in (12, 30):
        train_mean = float(train[epoch - 1])
        val_mean = float(val[epoch - 1])
        print(f"{epoch:4d}  {train_mean:8.4f}  {val_mean:8.4f}  {train_mean - val_mean:.3f}")


def check(rows, mark):
    ok = True
    print("[확인] 최고 검증 PR  vs  출력 3")
    for row in rows:
        good = same(row["best_val"], EXPECTED_BEST[row["seed"]], 4)
        ok = ok and good
        print(f"  시드 {row['seed']}  {row['best_val']:.4f}  기준 {EXPECTED_BEST[row['seed']]:.4f}  "
              f"{'일치' if good else '불일치'}")
    print("[확인] 30에폭 검증 PR  vs  출력 3")
    for row in rows:
        good = same(row["e30_val"], EXPECTED_E30_VAL[row["seed"]], 4)
        ok = ok and good
        print(f"  시드 {row['seed']}  {row['e30_val']:.4f}  기준 {EXPECTED_E30_VAL[row['seed']]:.4f}  "
              f"{'일치' if good else '불일치'}")
    print("[확인] 30에폭 학습 PR  vs  출력 3")
    for row in rows:
        good = same(row["e30_train"], EXPECTED_E30_TRAIN[row["seed"]], 3)
        ok = ok and good
        print(f"  시드 {row['seed']}  {row['e30_train']:.3f}  기준 {EXPECTED_E30_TRAIN[row['seed']]:.3f}  "
              f"{'일치' if good else '불일치'}")
    mark_text = f"{mark:.1f}"
    mark_ok = mark_text == EXPECTED_MARK
    ok = ok and mark_ok
    print(f"[확인] 시드별 최고 에폭 평균  {mark_text}  기준 {EXPECTED_MARK}  "
          f"{'일치' if mark_ok else '불일치'}")
    return ok


def _face(font_paths, weight, size):
    from matplotlib import font_manager

    prop = font_manager.FontProperties(fname=font_paths[weight])
    prop.set_size(size)
    return prop


def _axes(fig, font_paths):
    ax = fig.add_axes([0.12, 0.14, 0.82, 0.78])
    ax.set_facecolor(BG)
    ax.set_xlim(1, 30)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    tick_face = _face(font_paths, "bold", 12)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(tick_face)
        tick.set_color(TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E3E5EA")
    ax.spines["bottom"].set_color("#E3E5EA")
    ax.tick_params(color="#E3E5EA", length=0)
    label_face = _face(font_paths, "bold", 15)
    ax.set_xlabel("에폭", fontproperties=label_face, color="#4A4E55")
    return ax, label_face


def _legend(ax, font_paths):
    legend = ax.legend(loc="upper left", frameon=False, prop=_face(font_paths, "bold", 13))
    for text in legend.get_texts():
        text.set_color(TEXT)


def _save(fig, path):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    print(f"저장: {path}")


def plot_val(df, rows, mark, font_paths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, 31)
    val = series(df, "val_pr")
    mean = val.mean(axis=0)
    by_seed = {row["seed"]: row for row in rows}

    fig = plt.figure(figsize=(10.6, 5.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax, label_face = _axes(fig, font_paths)
    ax.axvspan(mark, 30, color="#D5D8DE", alpha=0.55, linewidth=0, zorder=0)
    ax.axvline(mark, color=MUTED, linestyle=":", linewidth=1.4, zorder=1)
    note = _face(font_paths, "extrabold", 13)
    ax.text((mark + 30) / 2, 0.96, "이후 개선 없음", transform=ax.get_xaxis_transform(),
            fontproperties=note, color=MUTED, ha="center", va="top")
    for index, seed in enumerate(SEEDS):
        ax.plot(epochs, val[index], color=SEED_COLOR[seed], linewidth=1.15,
                label=f"시드 {seed}", zorder=2)
        row = by_seed[seed]
        y = float(val[index, row["best_epoch"] - 1])
        ax.scatter([row["best_epoch"]], [y], s=42, color=SEED_COLOR[seed],
                   edgecolor="white", linewidth=1.1, zorder=5)
        ax.text(row["best_epoch"] + 0.35, y + 0.006, f"{seed}: {row['best_epoch']}에폭",
                fontproperties=note, color=TEXT, ha="left", va="bottom", zorder=6)
    ax.plot(epochs, mean, color=VAL_COLOR, linewidth=2.6, label="평균", zorder=3)
    ax.set_ylim(0.58, 0.74)
    ax.set_ylabel("검증 PR-AUC", fontproperties=label_face, color="#4A4E55", labelpad=8)
    _legend(ax, font_paths)
    _save(fig, OUT_VAL)
    plt.close(fig)


def _gap_arrow(ax, epoch, train_mean, val_mean, font, side):
    train_y = float(train_mean[epoch - 1])
    val_y = float(val_mean[epoch - 1])
    ax.annotate(
        "",
        xy=(epoch, train_y),
        xytext=(epoch, val_y),
        arrowprops=dict(arrowstyle="<->", color=TEXT, lw=1.2, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    x = epoch + 1.15 if side == "right" else epoch - 0.45
    ax.text(
        x, (train_y + val_y) / 2, f"차이 {train_y - val_y:.3f}",
        fontproperties=font, color=TEXT, ha="left" if side == "right" else "right",
        va="center", zorder=5,
    )


def plot_both(df, mark, font_paths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, 31)
    train = series(df, "train_pr")
    val = series(df, "val_pr")
    train_mean, val_mean = train.mean(axis=0), val.mean(axis=0)

    fig = plt.figure(figsize=(10.6, 5.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax, label_face = _axes(fig, font_paths)
    ax.fill_between(epochs, train.min(axis=0), train.max(axis=0),
                    color=TRAIN_COLOR, alpha=0.13, linewidth=0, zorder=0)
    ax.fill_between(epochs, val.min(axis=0), val.max(axis=0),
                    color=VAL_COLOR, alpha=0.13, linewidth=0, zorder=0)
    ax.plot(epochs, train_mean, color=TRAIN_COLOR, linewidth=2.4, linestyle="--",
            label="학습 PR-AUC", zorder=2)
    ax.plot(epochs, val_mean, color=VAL_COLOR, linewidth=2.4, linestyle="-",
            label="검증 PR-AUC", zorder=2)
    ax.axvline(mark, color=MUTED, linestyle=":", linewidth=1.4, zorder=1)
    note = _face(font_paths, "extrabold", 13)
    ax.text(mark, 0.97, f"최고 에폭 평균 {mark:.1f}", transform=ax.get_xaxis_transform(),
            fontproperties=note, color=MUTED, ha="center", va="top", zorder=5)
    _gap_arrow(ax, 12, train_mean, val_mean, note, side="right")
    _gap_arrow(ax, 30, train_mean, val_mean, note, side="left")
    ax.text(29.15, train_mean[-1], f"{train_mean[-1]:.3f}",
            fontproperties=note, color=TRAIN_COLOR, ha="right", va="bottom")
    ax.text(29.15, val_mean[-1], f"{val_mean[-1]:.3f}",
            fontproperties=note, color=VAL_COLOR, ha="right", va="top")
    ax.set_ylabel("PR-AUC", fontproperties=label_face, color="#4A4E55", labelpad=8)
    _legend(ax, font_paths)
    _save(fig, OUT_BOTH)
    plt.close(fig)


def main():
    df = load_history()
    if df is None:
        return 1
    rows = seed_rows(df)
    mark = float(np.mean([row["best_epoch"] for row in rows]))
    curve_epoch = mean_curve_best(series(df, "val_pr"))
    print_table(rows, mark, curve_epoch)
    print_gaps(df)
    if not check(rows, mark):
        print("[중단] 출력 3과 다릅니다. 그림은 저장하지 않습니다.")
        return 1
    font_paths = slides._gothic_paths()
    from matplotlib import font_manager
    for path in font_paths.values():
        font_manager.fontManager.addfont(path)
    plot_val(df, rows, mark, font_paths)
    plot_both(df, mark, font_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
