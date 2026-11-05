"""
4단계 배치 크기 (새 흐름).

출발: (28,56,28), Adam 0.01, 최대 30에폭, 드롭아웃 0.2, 조기종료 없음,
클래스 가중치, 시드 42/43/44.
학습·기록은 c1_flow_common만 쓴다. c1_flow_runs.csv에 있는 cfg+시드는 재사용한다.

사용법:
  python members/C1/c1_step4_batch.py
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

BATCHES = (32, 64, 128, 256)
CURRENT_BATCH = 128
EXPECTED_MEAN = 0.6919
NO_EPOCH = "기록 없음"
HEADERS = [
    "후보", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "양성예측",
    "최고/전체에폭", "실행", "30에폭 검증PR", "검증 하락", "학습 시간(초)",
]
ALIGNS = ["left"] + ["right"] * 7 + ["left"] + ["right"] * 3


def make_cfg(batch):
    return flow.default_cfg(
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


def render(rows):
    widths = [
        max(flow.disp_width(HEADERS[i]), *(flow.disp_width(row[i]) for row in rows))
        for i in range(len(HEADERS))
    ]
    lines = ["  ".join(flow.pad(HEADERS[i], widths[i], ALIGNS[i]) for i in range(len(HEADERS)))]
    for row in rows:
        lines.append("  ".join(flow.pad(row[i], widths[i], ALIGNS[i]) for i in range(len(row))))
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


def combo_row(label, frame, source):
    reports = flow.seed_reports(frame)
    cells, val_mean, val_std = flow.combo_cells(label, reports, max_epochs=30)
    cells = cells + [source] + overfit_extra(frame) + [seconds_of(frame)]
    return dict(
        label=label, cells=cells, val_mean=val_mean, val_std=val_std,
        reports=reports, frame=frame, source=source,
    )


def check_batch128(item):
    print("\n[확인] 배치 128 vs c1_flow_runs.csv 드롭아웃 0.2")
    ref_frames = []
    cfg = make_cfg(CURRENT_BATCH)
    for seed in flow.SEEDS:
        hit = flow.cached_rows(cfg, seed)
        if hit is None:
            print("  c1_flow_runs.csv에 드롭아웃 0.2 배치 128이 없습니다.")
            return
        ref_frames.append(hit)
    ref = flow.seed_reports(pd.concat(ref_frames, ignore_index=True))
    ref_mean = float(ref["val_pr"].mean())
    got_mean = item["val_mean"]
    print(f"  이번 평균 {got_mean:.4f}  csv {ref_mean:.4f}  기대 {EXPECTED_MEAN:.4f}  "
          f"{'일치' if round(got_mean, 4) == EXPECTED_MEAN and round(ref_mean, 4) == EXPECTED_MEAN else '불일치'}")
    for seed in flow.SEEDS:
        got = float(item["reports"].loc[item["reports"]["seed"] == seed, "val_pr"].iloc[0])
        expected = float(ref.loc[ref["seed"] == seed, "val_pr"].iloc[0])
        same = round(got, 6) == round(expected, 6)
        print(f"  seed {seed}  이번 {got:.6f}  csv {expected:.6f}  {'일치' if same else '불일치'}")


def main():
    t0 = time.perf_counter()
    d = flow.load_data()
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} (test 미사용)")
    print(flow.key_of(make_cfg(CURRENT_BATCH)))
    print("배치 32/64/128/256 × 시드 42/43/44, Adam 0.01, 드롭아웃 0.2, 클래스 가중치")

    cache = flow._read_cache()
    stats = []
    for batch in BATCHES:
        cfg = make_cfg(batch)
        src = source_of(cfg, cache)
        label = f"배치 {batch}"
        print(f"\n[조합] {label}  {src}", flush=True)
        frame = flow.run_seeds(cfg, d)
        stats.append(combo_row(label, frame, src))

    print()
    print("=== 배치 크기 ===")
    print("(28,56,28), Adam 0.01, 최대 30에폭, 드롭아웃 0.2, 조기종료 없음, 클래스 가중치")
    rows = [item["cells"] for item in stats]
    text, widths = render(rows)
    print(text)

    current = next(item for item in stats if item["label"] == f"배치 {CURRENT_BATCH}")
    top = max(stats, key=lambda item: item["val_mean"])
    diff = top["val_mean"] - current["val_mean"]
    tol = max(top["val_std"], current["val_std"])
    print("1위 후보")
    print("  ".join(flow.pad(top["cells"][i], widths[i], ALIGNS[i]) for i in range(len(HEADERS))))
    print(f"판정: 1위 {top['label']}. 현재 값(배치 {CURRENT_BATCH})과의 차이 {diff:+.4f}, "
          f"두 후보 표준편차 중 큰 값 {tol:.4f} -> 선택({top['label']})")

    check_batch128(current)
    print(f"\n저장: {flow.RUNS_CSV}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
