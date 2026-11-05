"""발표용 그림(히트맵, 학습 곡선, 오차 막대). 학습은 하지 않고 flow_core 결과만 그린다."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from flow_core import FIG_DIR, FLOW_DIR  # noqa: E402

BG = "#F5F6F7"
INK = "#31343A"
MUTED = "#6B6F76"
BLUE = "#2F5BD3"
ORANGE = "#D9711F"
YELLOW = "#FEF4A1"
CMAP = "Blues"


def _setup_font():
    for d in (os.path.join(FLOW_DIR, "fonts"), os.path.join(FLOW_DIR, "..", "figures", "fonts")):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith((".ttf", ".otf")):
                    font_manager.fontManager.addfont(os.path.join(d, f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Gothic A1", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"):
        if fam in names:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_font()


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"그림 저장: {path}")
    return path


def _axes(w=10, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9CCD1")
    ax.tick_params(colors=INK)
    return fig, ax


def heatmap(values, row_labels, col_labels, name, selected=None, best=None,
            xlabel="", ylabel="", sub=None):
    """values[i][j] = 평균 PR-AUC. selected=(i,j) 노란 칸, best=(i,j) 굵은 테두리. sub[i][j] 작은 글씨."""
    import numpy as np
    arr = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(1.9 * len(col_labels) + 2, 1.25 * len(row_labels) + 1.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    cmap = plt.get_cmap(CMAP)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            color = YELLOW if selected == (i, j) else cmap(0.12 + 0.55 * (v - vmin) / (vmax - vmin + 1e-12))
            lw = 4 if best == (i, j) else 2
            ax.add_patch(plt.Rectangle((j + 0.05, i + 0.05), 0.9, 0.9, facecolor=color,
                                       edgecolor=INK, linewidth=lw))
            ax.text(j + 0.5, i + (0.45 if sub else 0.5), f"{v:.3f}", ha="center", va="center",
                    fontsize=20, fontweight="bold", color=INK)
            if sub:
                ax.text(j + 0.5, i + 0.75, sub[i][j], ha="center", va="center", fontsize=10, color=MUTED)
    ax.set_xlim(0, arr.shape[1])
    ax.set_ylim(arr.shape[0], 0)
    ax.set_xticks([j + 0.5 for j in range(arr.shape[1])], col_labels, fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(arr.shape[0])], row_labels, fontsize=12)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color=INK)
    return _save(fig, name)


def mean_band(ax, df, col, color, label, ls="-"):
    g = df.groupby("epoch")[col]
    m, lo, hi = g.mean(), g.min(), g.max()
    ax.fill_between(m.index, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(m.index, m.values, color=color, lw=2.6, ls=ls, label=label)
    return m


def finish(ax, xlabel="에폭", ylabel="PR-AUC", legend=True):
    ax.set_xlabel(xlabel, fontsize=13, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, color=INK, fontweight="bold")
    if legend:
        leg = ax.legend(frameon=False, fontsize=12, loc="upper left")
        for t in leg.get_texts():
            t.set_color(INK)


def errorbars(results, name, selected_label, xlabel, sub_labels=None):
    fig, ax = _axes(9, 5)
    xs = list(range(len(results)))
    for x, r in zip(xs, results):
        color = BLUE if r["label"] == selected_label else "#9AA0A8"
        ax.errorbar(x, r["mean"], yerr=r["std"], fmt="o", ms=11, color=color, capsize=8, lw=2.5)
        for _, s in r["seeds"].iterrows():
            ax.plot(x + 0.12, s["val_pr"], "o", ms=5, color=color, alpha=0.35)
        ax.text(x + 0.18, r["mean"], f"{r['mean']:.4f}", va="center", fontsize=11, color=INK)
        if r["label"] == selected_label:
            ax.text(x, r["mean"] + r["std"] + 0.0025, "선택", ha="center", fontsize=12,
                    color=BLUE, fontweight="bold")
    labels = [r["label"] for r in results]
    if sub_labels:
        labels = [f"{a}\n{b}" for a, b in zip(labels, sub_labels)]
    ax.set_xticks(xs, labels, fontsize=11)
    ax.set_xlim(-0.5, len(results) - 0.3)
    lo = min(r["mean"] - r["std"] for r in results) - 0.006
    ax.set_ylim(lo, max(r["mean"] + r["std"] for r in results) + 0.008)
    ax.set_ylabel("검증 PR-AUC", fontsize=13, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel + f"   (y축 {lo:.2f}부터)", fontsize=12, color=INK)
    return _save(fig, name)
