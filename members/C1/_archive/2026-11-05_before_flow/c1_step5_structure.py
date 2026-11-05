"""
5단계 구조 확인 (새 흐름).

조건: Adam 0.01, 배치 64, 최대 30에폭, 드롭아웃 0.2(모든 은닉층),
조기종료 없음, 클래스 가중치, ReLU, he_normal, 시드 42/43/44.
학습·기록은 c1_flow_common만 쓴다. c1_flow_runs.csv에 있는 cfg+시드는 재사용한다.

사용법:
  python members/C1/c1_step5_structure.py
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

CURRENT = (28, 56, 28)
EXPECTED_MEAN = 0.6945
NO_EPOCH = "기록 없음"
CANDIDATES = [
    (1, (14,)),
    (1, (28,)),
    (1, (56,)),
    (2, (14, 28)),
    (2, (28, 56)),
    (2, (56, 112)),
    (3, (14, 28, 14)),
    (3, (28, 56, 28)),
    (3, (56, 112, 56)),
]
HEADERS = [
    "후보", "가중치 수", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "양성예측",
    "최고/전체에폭", "30에폭 검증PR", "검증 하락", "학습 시간",
]
ALIGNS = ["left"] + ["right"] * 11
SEED_HEADERS = ["후보", "시드", "검증PR", "학습PR", "차이", "최고에폭"]
SEED_ALIGNS = ["left"] + ["right"] * 5


def hidden_text(hidden):
    return "(" + ",".join(str(u) for u in hidden) + ")"


def make_cfg(hidden):
    return flow.default_cfg(
        hidden=tuple(hidden),
        dropout=0.2,
        drop_where="all",
        optimizer="Adam",
        lr=0.01,
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


def combo_row(n_layers, hidden, frame, source):
    label = f"{n_layers}층 {hidden_text(hidden)}"
    reports = flow.seed_reports(frame)
    cells, val_mean, val_std = flow.combo_cells(label, reports, max_epochs=30)
    cells = [cells[0], n_params_of(frame)] + cells[1:] + overfit_extra(frame) + [seconds_of(frame)]
    return dict(
        n_layers=n_layers, hidden=tuple(hidden), label=label, cells=cells,
        val_mean=val_mean, val_std=val_std, reports=reports, frame=frame, source=source,
    )


def check_current(item):
    print("\n[확인] (28,56,28) vs c1_step4_batch.py 배치 64")
    ref_frames = []
    cfg = make_cfg(CURRENT)
    for seed in flow.SEEDS:
        hit = flow.cached_rows(cfg, seed)
        if hit is None:
            print("  c1_flow_runs.csv에 배치 64 (28,56,28)이 없습니다.")
            return
        ref_frames.append(hit)
    ref = flow.seed_reports(pd.concat(ref_frames, ignore_index=True))
    ref_mean = float(ref["val_pr"].mean())
    got_mean = item["val_mean"]
    same_mean = round(got_mean, 4) == EXPECTED_MEAN and round(ref_mean, 4) == EXPECTED_MEAN
    print(f"  이번 평균 {got_mean:.4f}  csv {ref_mean:.4f}  기대 {EXPECTED_MEAN:.4f}  "
          f"{'일치' if same_mean else '불일치'}")
    for seed in flow.SEEDS:
        got = float(item["reports"].loc[item["reports"]["seed"] == seed, "val_pr"].iloc[0])
        expected = float(ref.loc[ref["seed"] == seed, "val_pr"].iloc[0])
        same = round(got, 6) == round(expected, 6)
        print(f"  seed {seed}  이번 {got:.6f}  csv {expected:.6f}  {'일치' if same else '불일치'}")


def main():
    t0 = time.perf_counter()
    d = flow.load_data()
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} (test 미사용)")
    print(flow.key_of(make_cfg(CURRENT)))
    print("1~3층 × 좁음/현재/넓음, 배치 64, Adam 0.01, 드롭아웃 0.2, 클래스 가중치")

    cache = flow._read_cache()
    stats = []
    for n_layers, hidden in CANDIDATES:
        cfg = make_cfg(hidden)
        src = source_of(cfg, cache)
        label = f"{n_layers}층 {hidden_text(hidden)}"
        print(f"\n[조합] {label}  {src}", flush=True)
        frame = flow.run_seeds(cfg, d)
        stats.append(combo_row(n_layers, hidden, frame, src))

    print()
    print("=== 구조 확인 (배치 64, 그 외 그대로) ===")
    print("(28,56,28) 기준, Adam 0.01, 배치 64, 최대 30에폭, 드롭아웃 0.2(모든 은닉층), "
          "조기종료 없음, 클래스 가중치, ReLU, he_normal")
    rows = [item["cells"] for item in stats]
    text, widths = render(HEADERS, ALIGNS, rows)
    print(text)

    print("층별 1위")
    for n_layers in (1, 2, 3):
        group = [item for item in stats if item["n_layers"] == n_layers]
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

    current = next(item for item in stats if item["hidden"] == CURRENT)
    top = max(stats, key=lambda item: item["val_mean"])
    diff = top["val_mean"] - current["val_mean"]
    tol = max(top["val_std"], current["val_std"])
    keep = top["hidden"] == CURRENT or diff <= tol
    flag = f"유지({current['label']})" if keep else f"변경({top['label']})"
    print(f"판정: 1위 {top['label']}. 현재 값 {hidden_text(CURRENT)}과의 차이 {diff:+.4f}, "
          f"두 후보 표준편차 중 큰 값 {tol:.4f} -> {flag}")

    check_current(current)
    print(f"\n저장: {flow.RUNS_CSV}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
