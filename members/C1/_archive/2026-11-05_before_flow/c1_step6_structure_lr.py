"""
6단계 구조 × 학습률 (새 흐름).

조건: Adam, 배치 64, 최대 30에폭, 드롭아웃 0.2(모든 은닉층),
조기종료 없음, 클래스 가중치, ReLU, he_normal, 시드 42/43/44.
학습·기록은 c1_flow_common만 쓴다. c1_flow_runs.csv에 있는 cfg+시드는 재사용한다.

사용법:
  python members/C1/c1_step6_structure_lr.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

C1_DIR = os.path.dirname(os.path.abspath(__file__))
if C1_DIR not in sys.path:
    sys.path.insert(0, C1_DIR)

import c1_flow_common as flow
import c1_slide_figures as slides

FIG_DIR = os.path.join(C1_DIR, "figures")
OUT_PNG = os.path.join(FIG_DIR, "structure_lr_heatmap.png")
HIDDENS = ((28, 56, 28), (56, 112, 56))
LRS = (0.0033, 0.01, 0.03)
LR_TEXT = {0.0033: "0.0033", 0.01: "0.01", 0.03: "0.03"}
EXPECTED = {
    ((28, 56, 28), 0.01): 0.6945,
    ((56, 112, 56), 0.01): 0.7016,
}
NO_EPOCH = "기록 없음"
HEADERS = [
    "후보", "가중치 수", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "양성예측",
    "최고/전체에폭", "30에폭 검증PR", "검증 하락", "학습 시간",
]
ALIGNS = ["left"] + ["right"] * 11
SEED_HEADERS = ["후보", "시드", "검증PR", "학습PR", "차이", "최고에폭"]
SEED_ALIGNS = ["left"] + ["right"] * 5


def hidden_text(hidden):
    return "(" + ",".join(str(u) for u in hidden) + ")"


def combo_label(hidden, lr):
    return f"{hidden_text(hidden)} {LR_TEXT[lr]}"


def make_cfg(hidden, lr):
    return flow.default_cfg(
        hidden=tuple(hidden),
        dropout=0.2,
        drop_where="all",
        optimizer="Adam",
        lr=float(lr),
        batch=64,
        max_epochs=30,
        early_stop=0,
        class_weight="balanced",
    )


def source_of(cfg, cache):
    hits = 0
    for seed in flow.SEEDS:
        hit = flow.cached_rows(cfg, seed, cache)
        if hit is not None and not hit.empty:
            hits += 1
    n = len(flow.SEEDS)
    if hits == n:
        return "재사용"
    if hits == 0:
        return "새로 실행"
    return f"재사용 {hits}/새로 실행 {n - hits}"


def render(headers, aligns, rows):
    widths = [
        max(flow.disp_width(headers[i]), *(flow.disp_width(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = ["  ".join(flow.pad(headers[i], widths[i], aligns[i]) for i in range(len(headers)))]
    for row in rows:
        lines.append("  ".join(flow.pad(row[i], widths[i], aligns[i]) for i in range(len(row))))
    return "\n".join(lines), widths


def overfit_extra(frame):
    e30, drops = [], []
    for seed in flow.SEEDS:
        sub = frame[frame["seed"] == seed]
        if sub.empty:
            return [NO_EPOCH, NO_EPOCH]
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


def seconds_of(frame):
    vals = []
    for seed in flow.SEEDS:
        sub = frame[frame["seed"] == seed]
        if sub.empty or "seconds" not in sub.columns:
            continue
        vals.append(float(sub.iloc[0]["seconds"]))
    if not vals:
        return "기록 없음"
    return f"{float(np.mean(vals)):.1f}"


def n_params_of(frame):
    return str(int(frame.iloc[0]["n_params"]))


def combo_row(hidden, lr, frame, source):
    label = combo_label(hidden, lr)
    reports = flow.seed_reports(frame)
    cells, val_mean, val_std = flow.combo_cells(label, reports, max_epochs=30)
    cells = [cells[0], n_params_of(frame)] + cells[1:] + overfit_extra(frame) + [seconds_of(frame)]
    return dict(
        hidden=tuple(hidden), lr=float(lr), label=label, cells=cells,
        val_mean=val_mean, val_std=val_std, reports=reports, frame=frame, source=source,
    )


def find_item(stats, hidden, lr):
    for item in stats:
        if item["hidden"] == tuple(hidden) and abs(item["lr"] - lr) < 1e-12:
            return item
    return None


def check_cached(stats):
    print("\n[확인] 학습률 0.01 vs c1_step5_structure.py")
    ok = True
    for hidden, lr in EXPECTED:
        item = find_item(stats, hidden, lr)
        want = EXPECTED[(hidden, lr)]
        cfg = make_cfg(hidden, lr)
        ref_frames = [flow.cached_rows(cfg, seed) for seed in flow.SEEDS]
        if item is None or any(hit is None for hit in ref_frames):
            print(f"  {combo_label(hidden, lr)}  없음")
            ok = False
            continue
        ref = flow.seed_reports(pd.concat(ref_frames, ignore_index=True))
        got_mean = item["val_mean"]
        ref_mean = float(ref["val_pr"].mean())
        same_mean = round(got_mean, 4) == want and round(ref_mean, 4) == want
        ok = ok and same_mean
        print(f"  {combo_label(hidden, lr)}  이번 {got_mean:.4f}  csv {ref_mean:.4f}  "
              f"기대 {want:.4f}  {'일치' if same_mean else '불일치'}")
        for seed in flow.SEEDS:
            got = float(item["reports"].loc[item["reports"]["seed"] == seed, "val_pr"].iloc[0])
            expected = float(ref.loc[ref["seed"] == seed, "val_pr"].iloc[0])
            same = round(got, 6) == round(expected, 6)
            ok = ok and same
            print(f"    seed {seed}  이번 {got:.6f}  csv {expected:.6f}  "
                  f"{'일치' if same else '불일치'}")
    return ok


def draw_heatmap(stats, selected):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    font_paths = slides._gothic_paths()
    for path in font_paths.values():
        font_manager.fontManager.addfont(path)

    def face(weight, size):
        prop = font_manager.FontProperties(fname=font_paths[weight])
        prop.set_size(size)
        return prop

    means = [item["val_mean"] for item in stats]
    lo, hi = min(means), max(means)
    span = max(hi - lo, 1e-6)
    cmap = slides._heat_colors()
    selected_key = (selected["hidden"], selected["lr"])

    fig = plt.figure(figsize=(10.6, 5.2), dpi=200)
    fig.patch.set_facecolor("#F5F6F7")
    ax = fig.add_axes([0.26, 0.16, 0.70, 0.70])
    ax.set_facecolor("#F5F6F7")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    pad_x, pad_y = 0.016, 0.040
    gap_x, gap_y = 0.020, 0.060
    cols, rows = len(LRS), len(HIDDENS)
    cell_w = (1 - 2 * pad_x - gap_x * (cols - 1)) / cols
    cell_h = (1 - 2 * pad_y - gap_y * (rows - 1)) / rows
    number_face = face("extrabold", 28)
    row_face = face("extrabold", 16)
    col_face = face("bold", 16)
    axis_face = face("bold", 18)

    by_key = {(item["hidden"], item["lr"]): item for item in stats}
    for row, hidden in enumerate(HIDDENS):
        y = pad_y + (rows - 1 - row) * (cell_h + gap_y)
        ax.text(
            -0.03, y + cell_h / 2, hidden_text(hidden),
            transform=ax.transAxes, ha="right", va="center",
            fontproperties=row_face, color="#31343A", clip_on=False,
        )
        for col, lr in enumerate(LRS):
            x = pad_x + col * (cell_w + gap_x)
            item = by_key[(hidden, lr)]
            value = item["val_mean"]
            if (hidden, lr) == selected_key:
                color = "#FEF4A1"
            else:
                tone = (value - lo) / span
                color = cmap(min(1.0, max(0.0, tone)))
            ax.add_patch(Rectangle(
                (x, y), cell_w, cell_h, transform=ax.transAxes,
                facecolor=color, edgecolor="#31343A",
                linewidth=3, joinstyle="miter", zorder=2,
            ))
            ax.text(
                x + cell_w / 2, y + cell_h / 2, f"{value:.4f}",
                transform=ax.transAxes, ha="center", va="center",
                fontproperties=number_face, color="#31343A", zorder=3,
            )
            if row == 0:
                ax.text(
                    x + cell_w / 2, 1.04, LR_TEXT[lr],
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontproperties=col_face, color="#31343A", clip_on=False,
                )

    ax.text(
        0.5, -0.12, "학습률",
        transform=ax.transAxes, ha="center", va="top",
        fontproperties=axis_face, color="#4A4E55", clip_on=False,
    )
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"저장: {OUT_PNG}")


def main():
    t0 = time.perf_counter()
    d = flow.load_data()
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} (test 미사용)")
    print("구조 (28,56,28)/(56,112,56) × 학습률 0.0033/0.01/0.03, 배치 64, 드롭아웃 0.2")

    cache = flow._read_cache()
    stats = []
    for hidden in HIDDENS:
        for lr in LRS:
            cfg = make_cfg(hidden, lr)
            src = source_of(cfg, cache)
            label = combo_label(hidden, lr)
            print(f"\n[조합] {label}  {src}", flush=True)
            frame = flow.run_seeds(cfg, d)
            stats.append(combo_row(hidden, lr, frame, src))

    print()
    print("=== 구조 × 학습률 (배치 64, 그 외 그대로) ===")
    print("Adam, 배치 64, 최대 30에폭, 드롭아웃 0.2(모든 은닉층), 조기종료 없음, "
          "클래스 가중치, ReLU, he_normal")
    rows = [item["cells"] for item in stats]
    text, widths = render(HEADERS, ALIGNS, rows)
    print(text)

    print("구조별 1위 학습률")
    for hidden in HIDDENS:
        group = [item for item in stats if item["hidden"] == hidden]
        top = max(group, key=lambda item: item["val_mean"])
        print("  ".join(flow.pad(top["cells"][i], widths[i], ALIGNS[i]) for i in range(len(HEADERS))))

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

    ranked = sorted(stats, key=lambda item: item["val_mean"], reverse=True)
    top, second = ranked[0], ranked[1]
    diff = top["val_mean"] - second["val_mean"]
    tol = max(top["val_std"], second["val_std"])
    print(f"판정: 전체 1위 {top['label']}. 2위 {second['label']}와의 차이 {diff:+.4f}, "
          f"두 후보 표준편차 중 큰 값 {tol:.4f} -> 선택({top['label']})")

    print("비교: 학습률마다 (56,112,56) − (28,56,28)")
    wide, paper = (56, 112, 56), (28, 56, 28)
    for lr in LRS:
        a = find_item(stats, wide, lr)
        b = find_item(stats, paper, lr)
        gap = a["val_mean"] - b["val_mean"]
        bigger = max(a["val_std"], b["val_std"])
        print(f"  lr {LR_TEXT[lr]}  {gap:+.4f}, 두 표준편차 중 큰 값 {bigger:.4f}")

    check_cached(stats)
    draw_heatmap(stats, top)
    print(f"저장: {flow.RUNS_CSV}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
