"""
c1_flow_runs.csv만 읽어 배치 크기 표와 그림을 만든다.
학습하지 않는다.

사용법:
  python members/C1/c1_step4_batch_figure.py
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata

import numpy as np
import pandas as pd

C1_DIR = os.path.dirname(os.path.abspath(__file__))
if C1_DIR not in sys.path:
    sys.path.insert(0, C1_DIR)

import c1_slide_figures as slides

RUNS_CSV = os.path.join(C1_DIR, "c1_flow_runs.csv")
FIG_DIR = os.path.join(C1_DIR, "figures")
OUT_PNG = os.path.join(FIG_DIR, "batch_pr_auc.png")
BATCHES = (32, 64, 128, 256)
CURRENT_BATCH = 128
SELECT_BATCH = 64
SEEDS = (42, 43, 44)
NO_EPOCH = "기록 없음"
EXPECTED = {
    32: "0.6891±0.0044",
    64: "0.6945±0.0074",
    128: "0.6919±0.0099",
    256: "0.6891±0.0062",
}
HEADERS = [
    "후보", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "양성예측",
    "최고/전체에폭", "30에폭 검증PR", "검증 하락", "학습 시간(초)",
]
ALIGNS = ["left"] + ["right"] * 10
SEED_HEADERS = ["후보", "시드", "검증PR", "학습PR", "차이", "최고에폭"]
SEED_ALIGNS = ["left"] + ["right"] * 5
TEXT = "#31343A"
MUTED = "#6A6E75"
BG = "#F5F6F7"
SELECT_COLOR = "#2F5BD8"
GRAY = "#9AA0A8"


def make_cfg(batch):
    return dict(
        hidden=(28, 56, 28),
        dropout=0.2,
        drop_where="all",
        optimizer="Adam",
        lr=0.01,
        batch=int(batch),
        max_epochs=30,
        early_stop=0,
        class_weight="balanced",
    )


def key_of(cfg):
    payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(cfg.items())}
    return json.dumps(payload)


def disp_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def pad(text, width, align="left"):
    gap = width - disp_width(text)
    if gap <= 0:
        return text
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def render(headers, aligns, rows):
    widths = [
        max(disp_width(headers[i]), *(disp_width(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = ["  ".join(pad(headers[i], widths[i], aligns[i]) for i in range(len(headers)))]
    for row in rows:
        lines.append("  ".join(pad(row[i], widths[i], aligns[i]) for i in range(len(row))))
    return "\n".join(lines), widths


def seed_reports(frame):
    rows = []
    for seed, sub in frame.groupby("seed"):
        row = sub.iloc[0]
        rows.append(dict(
            seed=int(seed),
            val_pr=float(row["report_val_pr"]),
            train_pr=float(row["report_train_pr"]),
            recall=float(row["report_recall"]),
            precision=float(row["report_precision"]),
            pos_rate=float(row["report_pos_rate"]),
            best_epoch=int(row["best_epoch"]),
            n_epochs=int(row["n_epochs"]),
        ))
    return pd.DataFrame(rows).sort_values("seed")


def combo_cells(label, reports):
    val = reports["val_pr"].to_numpy(dtype=float)
    train = reports["train_pr"].to_numpy(dtype=float)
    val_mean, train_mean = float(val.mean()), float(train.mean())
    val_std = float(val.std(ddof=1))
    cells = [
        label,
        f"{val_mean:.4f}±{val_std:.4f}",
        f"{train_mean:.3f}",
        f"{train_mean - val_mean:+.3f}",
        f"{float(reports['recall'].mean()):.3f}",
        f"{float(reports['precision'].mean()):.4f}",
        f"{float(reports['pos_rate'].mean()):.3f}",
        f"{int(round(float(reports['best_epoch'].mean())))}/"
        f"{int(round(float(reports['n_epochs'].mean())))}",
    ]
    return cells, val_mean, val_std


def load_combo(df, batch):
    key = key_of(make_cfg(batch))
    hit = df[df["cfg"] == key]
    parts = []
    for seed in SEEDS:
        one = hit[hit["seed"] == seed]
        if one.empty:
            return None
        parts.append(one.sort_values("epoch"))
    return pd.concat(parts, ignore_index=True)


def overfit_extra(frame):
    e30, drops = [], []
    for seed in SEEDS:
        sub = frame[frame["seed"] == seed]
        best = int(sub.iloc[0]["best_epoch"])
        at_best = sub[sub["epoch"] == best]
        at_30 = sub[sub["epoch"] == 30]
        if at_best.empty or at_30.empty:
            return [NO_EPOCH, NO_EPOCH]
        e30.append(float(at_30.iloc[0]["val_pr_sk"]))
        drops.append(float(at_best.iloc[0]["val_pr_sk"]) - float(at_30.iloc[0]["val_pr_sk"]))
    e30 = np.array(e30, dtype=float)
    return [
        f"{float(e30.mean()):.4f}±{float(e30.std(ddof=1)):.4f}",
        f"{float(np.mean(drops)):+.4f}",
    ]


def seconds_mean(frame):
    vals = [float(frame[frame["seed"] == seed].iloc[0]["seconds"]) for seed in SEEDS]
    return float(np.mean(vals))


def collect(df):
    stats = []
    missing = []
    for batch in BATCHES:
        frame = load_combo(df, batch)
        label = f"배치 {batch}"
        if frame is None:
            missing.append(label)
            continue
        reports = seed_reports(frame)
        cells, val_mean, val_std = combo_cells(label, reports)
        cells = cells + overfit_extra(frame) + [f"{seconds_mean(frame):.1f}"]
        seeds = [
            float(reports.loc[reports["seed"] == seed, "val_pr"].iloc[0])
            for seed in SEEDS
        ]
        stats.append(dict(
            batch=batch, label=label, cells=cells,
            val_mean=val_mean, val_std=val_std,
            reports=reports, seconds=seconds_mean(frame), seeds=seeds,
            text=f"{val_mean:.4f}±{val_std:.4f}",
        ))
    return missing, stats


def print_tables(stats):
    print("=== 배치 크기 (드롭아웃 0.2, 그 외 그대로) ===")
    print("(28,56,28), Adam 0.01, 최대 30에폭, 드롭아웃 0.2, 조기종료 없음, 클래스 가중치")
    rows = [item["cells"] for item in stats]
    text, _widths = render(HEADERS, ALIGNS, rows)
    print(text)
    seed_rows = []
    for item in stats:
        for _, row in item["reports"].iterrows():
            val = float(row["val_pr"])
            train = float(row["train_pr"])
            seed_rows.append([
                item["label"], str(int(row["seed"])),
                f"{val:.4f}", f"{train:.3f}", f"{train - val:+.3f}",
                str(int(row["best_epoch"])),
            ])
    print(render(SEED_HEADERS, SEED_ALIGNS, seed_rows)[0])
    current = next(item for item in stats if item["batch"] == CURRENT_BATCH)
    top = max(stats, key=lambda item: item["val_mean"])
    diff = top["val_mean"] - current["val_mean"]
    tol = max(top["val_std"], current["val_std"])
    print(f"판정: 1위 {top['label']}. 현재 값(배치 {CURRENT_BATCH})과의 차이 {diff:+.4f}, "
          f"두 후보 표준편차 중 큰 값 {tol:.4f} -> 선택({top['label']})")
    return top


def _face(font_paths, weight, size):
    from matplotlib import font_manager

    prop = font_manager.FontProperties(fname=font_paths[weight])
    prop.set_size(size)
    return prop


def plot(stats):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    font_paths = slides._gothic_paths()
    for path in font_paths.values():
        font_manager.fontManager.addfont(path)
    label_face = _face(font_paths, "bold", 15)
    tick_face = _face(font_paths, "bold", 13)
    value_face = _face(font_paths, "extrabold", 12)
    pick_face = _face(font_paths, "extrabold", 14)

    xs = np.arange(len(stats))
    means = np.array([item["val_mean"] for item in stats], dtype=float)
    stds = np.array([item["val_std"] for item in stats], dtype=float)
    seed_vals = np.array([item["seeds"] for item in stats], dtype=float)
    colors = [SELECT_COLOR if item["batch"] == SELECT_BATCH else GRAY for item in stats]

    all_vals = np.concatenate([seed_vals.ravel(), means - stds, means + stds])
    span = float(all_vals.max() - all_vals.min())
    pad = max(0.006, span * 0.35)
    y0 = float(all_vals.min()) - pad
    y1 = float(all_vals.max()) + pad * 1.35

    fig = plt.figure(figsize=(10.6, 5.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.12, 0.26, 0.82, 0.66])
    ax.set_facecolor(BG)

    for i, item in enumerate(stats):
        ax.scatter(
            [xs[i]] * len(SEEDS), item["seeds"],
            s=28, color=colors[i], alpha=0.28, zorder=3, linewidths=0,
        )
        ax.errorbar(
            xs[i], means[i], yerr=stds[i],
            fmt="o", color=colors[i], ecolor=colors[i],
            elinewidth=1.8, capsize=5, capthick=1.6,
            markersize=8.5, markeredgecolor="white", markeredgewidth=1.1,
            zorder=4,
        )
        ax.text(
            xs[i] + 0.12, means[i], f"{means[i]:.4f}",
            fontproperties=value_face, color=colors[i], ha="left", va="center",
        )
        if item["batch"] == SELECT_BATCH:
            ax.text(
                xs[i], means[i] + stds[i] + (y1 - y0) * 0.045, "선택",
                fontproperties=pick_face, color=SELECT_COLOR, ha="center", va="bottom",
            )

    ax.set_xlim(-0.55, len(stats) - 0.35)
    ax.set_ylim(y0, y1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(item["batch"]) for item in stats])
    ax.set_ylabel("검증 PR-AUC", fontproperties=label_face, color="#4A4E55", labelpad=10)
    ax.set_xlabel("")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(tick_face)
        tick.set_color(TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E3E5EA")
    ax.spines["bottom"].set_color("#E3E5EA")
    ax.tick_params(color="#E3E5EA", length=0)
    ax.yaxis.grid(True, color="#E3E5EA", linewidth=1.0)
    ax.set_axisbelow(True)

    for i, item in enumerate(stats):
        ax.text(
            xs[i], -0.09, f"{item['seconds']:.1f}초",
            transform=ax.get_xaxis_transform(),
            fontproperties=_face(font_paths, "bold", 11),
            color=MUTED, ha="center", va="top", clip_on=False,
        )
    ax.text(
        0.5, -0.20, "배치 크기",
        transform=ax.transAxes,
        fontproperties=label_face, color="#4A4E55",
        ha="center", va="top", clip_on=False,
    )
    ax.text(
        0.0, -0.32, f"y축 {y0:.2f}부터",
        transform=ax.transAxes,
        fontproperties=_face(font_paths, "bold", 12),
        color=MUTED, ha="left", va="top", clip_on=False,
    )

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"저장: {OUT_PNG}")
    return means, stds


def check(stats, fig_means, fig_stds):
    print("\n[확인] 터미널 요약 표·그림 vs c1_step4_batch.py")
    ok = True
    for i, item in enumerate(stats):
        table = item["text"]
        figure = f"{float(fig_means[i]):.4f}±{float(fig_stds[i]):.4f}"
        want = EXPECTED[item["batch"]]
        same = table == figure == want
        ok = ok and same
        print(f"  {item['label']}  표 {table}  그림 {figure}  기준 {want}  "
              f"{'일치' if same else '불일치'}")
    print("일치" if ok else "불일치")
    return ok


def main():
    if not os.path.isfile(RUNS_CSV):
        print(f"[중단] 없음: {RUNS_CSV}")
        return 1
    df = pd.read_csv(RUNS_CSV)
    missing, stats = collect(df)
    if missing:
        for label in missing:
            print(f"{label}  없음")
        return 1
    print_tables(stats)
    print()
    fig_means, fig_stds = plot(stats)
    return 0 if check(stats, fig_means, fig_stds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
