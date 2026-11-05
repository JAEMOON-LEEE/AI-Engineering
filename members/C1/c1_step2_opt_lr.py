"""
2단계 옵티마이저 × 학습률 (새 흐름).

학습·기록은 c1_flow_common만 쓴다. c1_flow_runs.csv에 있는 cfg+시드는 재사용한다.
최고 에폭은 Keras val_pr_auc, 보고 점수는 그 가중치의 sklearn AP.

사용법:
  python members/C1/c1_step2_opt_lr.py
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time

import pandas as pd

C1_DIR = os.path.dirname(os.path.abspath(__file__))
if C1_DIR not in sys.path:
    sys.path.insert(0, C1_DIR)

import c1_flow_common as flow
import c1_slide_figures as slides
import c1_ver2_adam as v2

FIG_DIR = os.path.join(C1_DIR, "figures")
OUT_PNG = os.path.join(FIG_DIR, "opt_lr_flow_heatmap.png")
SELECTED = ("Adam", 0.01)

OPTIMIZERS = ["SGD", "RMSprop", "Adam"]
LRS = [0.00005, 0.001, 0.01, 0.1]
LR_TEXT = {0.00005: "0.00005", 0.001: "0.001", 0.01: "0.01", 0.1: "0.1"}
OPT_LABEL = {"SGD": "SGD+NAG", "RMSprop": "RMSprop", "Adam": "Adam"}
HEADERS = flow.HEADERS + ["실행"]
ALIGNS = flow.ALIGNS + ["left"]


def make_cfg(optimizer, lr):
    return flow.default_cfg(
        hidden=(28, 56, 28),
        dropout=0.0,
        drop_where="none",
        optimizer=optimizer,
        lr=lr,
        batch=128,
        max_epochs=30,
        early_stop=0,
        class_weight="balanced",
    )


def combo_label(optimizer, lr, sgd_only=False):
    if sgd_only:
        return f"lr={LR_TEXT[lr]}"
    return f"{OPT_LABEL[optimizer]} {LR_TEXT[lr]}"


def source_of(cfg, cache):
    hits = 0
    for seed in flow.SEEDS:
        hit = flow.cached_rows(cfg, seed, cache)
        if hit is not None and not hit.empty:
            hits += 1
    if hits == len(flow.SEEDS):
        return "재사용"
    if hits == 0:
        return "새로 실행"
    return f"재사용 {hits}/새로 실행 {len(flow.SEEDS) - hits}"


def _as_float(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    return float(value)


def sgd_same_as_ver2():
    """ver2 실행 경로와 flow 실행 경로가 같은 SGD(모멘텀, NAG)인지."""
    opt = v2.make_optimizer("SGD", 0.01)
    momentum = _as_float(opt.momentum)
    nesterov = bool(opt.nesterov)
    ver2_calls = "make_optimizer(" in inspect.getsource(v2.build)
    flow_calls = "v2.make_optimizer(" in inspect.getsource(flow.build)
    same = ver2_calls and flow_calls and nesterov is True and abs(momentum - 0.9) < 1e-12
    return dict(same=same, momentum=momentum, nesterov=nesterov,
                ver2_calls=ver2_calls, flow_calls=flow_calls)


def official_rows(optimizer, lr):
    runs = pd.read_csv(v2.RUNS_CSV)

    def ok(raw):
        cfg = json.loads(raw)
        return (
            cfg.get("optimizer") == optimizer
            and abs(float(cfg.get("lr")) - lr) < 1e-12
            and cfg.get("hidden") == [28, 56, 28]
            and cfg.get("imbalance") == "class_weight"
            and cfg.get("batch") == 128
            and float(cfg.get("dropout", 0)) == 0.0
            and cfg.get("max_epochs") == 30
            and cfg.get("early_stop") == 0
        )

    hit = runs[runs["cfg"].map(ok)].drop_duplicates("seed")
    return hit.sort_values("seed")


def render(rows):
    widths = [
        max(flow.disp_width(HEADERS[i]), *(flow.disp_width(row[i]) for row in rows))
        for i in range(len(HEADERS))
    ]
    lines = ["  ".join(flow.pad(HEADERS[i], widths[i], ALIGNS[i]) for i in range(len(HEADERS)))]
    for row in rows:
        lines.append("  ".join(flow.pad(row[i], widths[i], ALIGNS[i]) for i in range(len(row))))
    return "\n".join(lines), widths


def format_row(cells, widths):
    return "  ".join(flow.pad(cells[i], widths[i], ALIGNS[i]) for i in range(len(cells)))


def run_grid(d):
    cache = flow._read_cache()
    frames = {}
    sources = {}
    for optimizer in OPTIMIZERS:
        for lr in LRS:
            cfg = make_cfg(optimizer, lr)
            sources[(optimizer, lr)] = source_of(cfg, cache)
            print(f"\n[조합] {combo_label(optimizer, lr)}  {sources[(optimizer, lr)]}", flush=True)
            frames[(optimizer, lr)] = flow.run_seeds(cfg, d)
    return frames, sources


def stats_of(frames, sources):
    stats = []
    for optimizer in OPTIMIZERS:
        for lr in LRS:
            reports = flow.seed_reports(frames[(optimizer, lr)])
            cells, val_mean, _std = flow.combo_cells(
                combo_label(optimizer, lr), reports, max_epochs=30,
            )
            cells.append(sources[(optimizer, lr)])
            stats.append(dict(
                optimizer=optimizer, lr=lr, val_mean=val_mean,
                cells=cells, reports=reports,
            ))
    return stats


def print_sgd(stats):
    rows = []
    for item in stats:
        if item["optimizer"] != "SGD":
            continue
        cells = list(item["cells"])
        cells[0] = combo_label("SGD", item["lr"], sgd_only=True)
        rows.append(cells)
    print("=== 학습률 그리드 (SGD+NAG, 그 외 0단계 그대로) ===")
    print("(28,56,28), SGD+NAG, 배치 128, 최대 30에폭, 조기종료 없음, 클래스 가중치")
    text, _widths = render(rows)
    print(text)


def winners(stats):
    picked = []
    for optimizer in OPTIMIZERS:
        group = [item for item in stats if item["optimizer"] == optimizer]
        picked.append(max(group, key=lambda item: item["val_mean"]))
    return picked


def print_grid(stats):
    rows = [item["cells"] for item in stats]
    print("=== 옵티마이저 × 학습률 그리드 (그 외 0단계 그대로) ===")
    print("(28,56,28), 배치 128, 최대 30에폭, 조기종료 없음, 클래스 가중치")
    text, widths = render(rows)
    print(text)
    print("옵티마이저별 1위 학습률")
    for item in winners(stats):
        print(format_row(item["cells"], widths))


def print_sgd_check(spec):
    print("[SGD 설정] c1_ver2_adam.make_optimizer")
    print(f"  SGD momentum={spec['momentum']}  nesterov={spec['nesterov']}")
    print(f"  ver2.build가 make_optimizer를 호출: {spec['ver2_calls']}")
    print(f"  flow.build가 v2.make_optimizer를 호출: {spec['flow_calls']}")
    if spec["same"]:
        print("  ver2_adam_runs.csv의 SGD와 NAG·모멘텀이 같습니다. 겹치는 SGD 조합을 비교합니다.")
    else:
        print("  ver2_adam_runs.csv의 SGD와 NAG·모멘텀이 다릅니다. 겹치는 SGD 조합은 비교에서 뺍니다.")


def print_repro(stats, include_sgd):
    print("\n[재현 확인] ver2_adam_runs.csv와 겹치는 조합, 시드별 val_pr")
    for item in stats:
        optimizer, lr = item["optimizer"], item["lr"]
        if optimizer == "SGD" and not include_sgd:
            print(f"\n{combo_label(optimizer, lr)}  제외 (SGD NAG·모멘텀이 ver2와 다름)")
            continue
        official = official_rows(optimizer, lr)
        if official.empty:
            print(f"\n{combo_label(optimizer, lr)}  겹치지 않음")
            continue
        print(f"\n{combo_label(optimizer, lr)}")
        print(f"{'seed':>6} {'이번 val_pr':>14} {'csv val_pr':>14} {'차이':>12}  결과")
        reports = item["reports"]
        for seed in flow.SEEDS:
            got_row = reports.loc[reports["seed"] == seed]
            ref = official.loc[official["seed"] == seed]
            if got_row.empty or ref.empty:
                print(f"{seed:6d}  시드 행 없음")
                continue
            got = float(got_row["val_pr"].iloc[0])
            expected = float(ref["val_pr"].iloc[0])
            same = round(got, 6) == round(expected, 6)
            print(f"{seed:6d} {got:14.6f} {expected:14.6f} {got - expected:12.6f}  "
                  f"{'일치' if same else '불일치'}")


def draw_heatmap(stats):
    if slides.SELECTED != SELECTED:
        raise SystemExit(
            f"히트맵 선택 칸이 다릅니다. 스크립트 {SELECTED}, 그림 모듈 {slides.SELECTED}"
        )
    heat = {
        (item["optimizer"], item["lr"]): {"mean": item["val_mean"]}
        for item in stats
    }
    os.makedirs(FIG_DIR, exist_ok=True)
    slides._draw_heatmap(heat, slides._gothic_paths(), OUT_PNG)
    print(f"저장: {OUT_PNG}")


def main():
    t0 = time.perf_counter()
    spec = sgd_same_as_ver2()
    print_sgd_check(spec)
    d = flow.load_data()
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} (test 미사용)")
    print("12조합 × 시드 42/43/44, 배치 128, 30에폭, 드롭아웃 0, 클래스 가중치")
    frames, sources = run_grid(d)
    stats = stats_of(frames, sources)
    print_repro(stats, include_sgd=spec["same"])
    print()
    print_sgd(stats)
    print()
    print_grid(stats)
    draw_heatmap(stats)
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
