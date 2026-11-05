"""
c1_step2_opt_lr.py 결과를 c1_flow_runs.csv만 읽어 출력한다.
학습하지 않는다. 없는 조합은 "없음"으로 표시하고 끝낸다.

사용법:
  python members/C1/c1_step2_print.py
"""

from __future__ import annotations

import json
import os
import unicodedata

import numpy as np
import pandas as pd

C1_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_CSV = os.path.join(C1_DIR, "c1_flow_runs.csv")
SEEDS = (42, 43, 44)
OPTIMIZERS = ["SGD", "RMSprop", "Adam"]
LRS = [0.00005, 0.001, 0.01, 0.1]
LR_TEXT = {0.00005: "0.00005", 0.001: "0.001", 0.01: "0.01", 0.1: "0.1"}
OPT_LABEL = {"SGD": "SGD+NAG", "RMSprop": "RMSprop", "Adam": "Adam"}
HEADERS = ["후보", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "양성예측", "최고/전체에폭", "실행"]
ALIGNS = ["left"] + ["right"] * 7 + ["left"]
ADAM_HEADERS = [
    "후보", "PR-AUC", "학습PR", "차이", "최고/전체에폭",
    "30에폭 검증PR", "30에폭 학습PR", "검증 하락", "학습 상승",
]
ADAM_ALIGNS = ["left"] + ["right"] * 8
SEED_HEADERS = [
    "후보", "시드", "검증PR", "학습PR", "차이", "최고에폭",
    "30에폭 검증PR", "30에폭 학습PR", "검증 하락", "학습 상승",
]
SEED_ALIGNS = ["left"] + ["right"] * 9
NO_EPOCH = "기록 없음"
# c1_step2_opt_lr.py 실행 때 출력 2의 숫자
EXPECTED = [
    ("SGD+NAG 0.00005", "0.3432±0.0742", "0.346", "+0.003", "0.672", "0.2552", "0.417", "30/30"),
    ("SGD+NAG 0.001", "0.6005±0.0093", "0.671", "+0.070", "0.747", "0.4550", "0.257", "30/30"),
    ("SGD+NAG 0.01", "0.6750±0.0052", "0.778", "+0.103", "0.801", "0.4508", "0.278", "22/30"),
    ("SGD+NAG 0.1", "0.6783±0.0133", "0.788", "+0.110", "0.827", "0.4144", "0.313", "20/30"),
    ("RMSprop 0.00005", "0.5355±0.0347", "0.573", "+0.038", "0.728", "0.3791", "0.302", "30/30"),
    ("RMSprop 0.001", "0.6592±0.0062", "0.782", "+0.123", "0.805", "0.4449", "0.283", "25/30"),
    ("RMSprop 0.01", "0.6827±0.0058", "0.768", "+0.086", "0.799", "0.4742", "0.265", "7/30"),
    ("RMSprop 0.1", "0.6463±0.0255", "0.696", "+0.049", "0.749", "0.5284", "0.227", "20/30"),
    ("Adam 0.00005", "0.5339±0.0429", "0.568", "+0.035", "0.735", "0.3796", "0.305", "30/30"),
    ("Adam 0.001", "0.6661±0.0195", "0.792", "+0.126", "0.803", "0.4439", "0.283", "26/30"),
    ("Adam 0.01", "0.6907±0.0097", "0.784", "+0.093", "0.812", "0.4433", "0.287", "12/30"),
    ("Adam 0.1", "0.6342±0.0110", "0.693", "+0.059", "0.738", "0.5131", "0.227", "7/30"),
]
CONDITION = "(28,56,28), 배치 128, 최대 30에폭, 드롭아웃 0, 조기종료 없음, 클래스 가중치"


def default_cfg(optimizer, lr):
    return dict(
        hidden=(28, 56, 28),
        dropout=0.0,
        drop_where="none",
        optimizer=optimizer,
        lr=float(lr),
        batch=128,
        max_epochs=30,
        early_stop=0,
        class_weight="balanced",
    )


def key_of(cfg):
    payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(cfg.items())}
    return json.dumps(payload)


def disp_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def pad(text, width, align):
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


def combo_label(optimizer, lr, sgd_only=False):
    if sgd_only:
        return f"lr={LR_TEXT[lr]}"
    return f"{OPT_LABEL[optimizer]} {LR_TEXT[lr]}"


def load_combo(df, optimizer, lr):
    key = key_of(default_cfg(optimizer, lr))
    hit = df[(df["cfg"] == key)]
    frames = []
    for seed in SEEDS:
        one = hit[hit["seed"] == seed]
        if one.empty:
            return None
        frames.append(one)
    return pd.concat(frames, ignore_index=True)


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
    cells = [
        label,
        f"{val_mean:.4f}±{float(val.std(ddof=1)):.4f}",
        f"{train_mean:.3f}",
        f"{train_mean - val_mean:+.3f}",
        f"{float(reports['recall'].mean()):.3f}",
        f"{float(reports['precision'].mean()):.4f}",
        f"{float(reports['pos_rate'].mean()):.3f}",
        f"{int(round(float(reports['best_epoch'].mean())))}/{int(round(float(reports['n_epochs'].mean())))}",
        "재사용",
    ]
    return cells, val_mean


def collect(df):
    missing = []
    stats = []
    for optimizer in OPTIMIZERS:
        for lr in LRS:
            frame = load_combo(df, optimizer, lr)
            label = combo_label(optimizer, lr)
            if frame is None:
                missing.append(label)
                continue
            reports = seed_reports(frame)
            cells, val_mean = combo_cells(label, reports)
            stats.append(dict(
                optimizer=optimizer, lr=lr, val_mean=val_mean,
                cells=cells, reports=reports, epochs=epoch_points(frame),
            ))
    return missing, stats


def print_sgd(stats):
    rows = []
    for item in stats:
        if item["optimizer"] != "SGD":
            continue
        cells = list(item["cells"])
        cells[0] = combo_label("SGD", item["lr"], sgd_only=True)
        rows.append(cells)
    print("=== 학습률 그리드 (SGD+NAG, 그 외 0단계 그대로) ===")
    print(f"(28,56,28), SGD+NAG, 배치 128, 최대 30에폭, 드롭아웃 0, 조기종료 없음, 클래스 가중치")
    print(render(HEADERS, ALIGNS, rows)[0])


def winners(stats):
    picked = []
    for optimizer in OPTIMIZERS:
        group = [item for item in stats if item["optimizer"] == optimizer]
        picked.append(max(group, key=lambda item: item["val_mean"]))
    return picked


def print_grid(stats):
    rows = [item["cells"] for item in stats]
    print("=== 옵티마이저 × 학습률 그리드 (그 외 0단계 그대로) ===")
    print(CONDITION)
    text, widths = render(HEADERS, ALIGNS, rows)
    print(text)
    print("옵티마이저별 1위 학습률")
    for item in winners(stats):
        print("  ".join(pad(item["cells"][i], widths[i], ALIGNS[i]) for i in range(len(HEADERS))))


def epoch_points(frame):
    """시드별 30에폭·최고에폭의 저장 기록. 둘 중 하나라도 없으면 None."""
    points = {}
    for seed, sub in frame.groupby("seed"):
        best = int(sub.iloc[0]["best_epoch"])
        at_best = sub[sub["epoch"] == best]
        at_30 = sub[sub["epoch"] == 30]
        if at_best.empty or at_30.empty:
            points[int(seed)] = None
            continue
        best_row, row30 = at_best.iloc[0], at_30.iloc[0]
        vals = [
            best_row["val_pr_sk"], best_row["train_pr"],
            row30["val_pr_sk"], row30["train_pr"],
        ]
        if not all(np.isfinite(float(v)) for v in vals):
            points[int(seed)] = None
            continue
        points[int(seed)] = dict(
            e30_val=float(row30["val_pr_sk"]),
            e30_train=float(row30["train_pr"]),
            best_val=float(best_row["val_pr_sk"]),
            best_train=float(best_row["train_pr"]),
        )
    return points


def overfit_cells(points):
    """평균 4열. 시드 하나라도 에폭 기록이 없으면 계산하지 않는다."""
    if any(points.get(seed) is None for seed in SEEDS):
        return [NO_EPOCH] * 4
    e30_val = np.array([points[s]["e30_val"] for s in SEEDS])
    e30_train = np.array([points[s]["e30_train"] for s in SEEDS])
    drop = np.array([points[s]["best_val"] - points[s]["e30_val"] for s in SEEDS])
    rise = np.array([points[s]["e30_train"] - points[s]["best_train"] for s in SEEDS])
    return [
        f"{float(e30_val.mean()):.4f}±{float(e30_val.std(ddof=1)):.4f}",
        f"{float(e30_train.mean()):.3f}",
        f"{float(drop.mean()):+.4f}",
        f"{float(rise.mean()):+.3f}",
    ]


def overfit_seed_cells(point):
    if point is None:
        return [NO_EPOCH] * 4
    return [
        f"{point['e30_val']:.4f}",
        f"{point['e30_train']:.3f}",
        f"{point['best_val'] - point['e30_val']:+.4f}",
        f"{point['e30_train'] - point['best_train']:+.3f}",
    ]


def print_adam(stats):
    adam = [item for item in stats if item["optimizer"] == "Adam"]
    recorded = [item["cells"][0] for item in adam if all(item["epochs"].get(s) is not None for s in SEEDS)]
    missing_epoch = [item["cells"][0] for item in adam if item["cells"][0] not in recorded]
    if recorded:
        print("에폭별 기록 있음: " + ", ".join(recorded))
    if missing_epoch:
        print("에폭별 기록 없음: " + ", ".join(missing_epoch))
    summary = [
        item["cells"][:4] + [item["cells"][7]] + overfit_cells(item["epochs"])
        for item in adam
    ]
    print("=== 과적합 확인 (Adam 4개 학습률) ===")
    print("(28,56,28), Adam, 배치 128, 최대 30에폭, 드롭아웃 0, 클래스 가중치")
    print(render(ADAM_HEADERS, ADAM_ALIGNS, summary)[0])
    seed_rows = []
    for item in adam:
        for _, row in item["reports"].iterrows():
            val = float(row["val_pr"])
            train = float(row["train_pr"])
            seed = int(row["seed"])
            seed_rows.append([
                item["cells"][0],
                str(seed),
                f"{val:.4f}",
                f"{train:.3f}",
                f"{train - val:+.3f}",
                str(int(row["best_epoch"])),
                *overfit_seed_cells(item["epochs"].get(seed)),
            ])
    print(render(SEED_HEADERS, SEED_ALIGNS, seed_rows)[0])
    print(
        "차이 = 학습PR − 검증PR, 최고에폭 = 검증 PR-AUC가 가장 높았던 에폭, "
        "검증 하락 > 0 이고 학습 상승 > 0 이면 최고 에폭 이후 과적합"
    )


def check(stats):
    ok = True
    for item, expected in zip(stats, EXPECTED):
        got = tuple(item["cells"][:8])
        if got != expected:
            ok = False
            print(f"[불일치] {item['cells'][0]}  이번 {' '.join(got[1:])}  기준 {' '.join(expected[1:])}")
    print("일치" if ok else "불일치")
    return ok


def main():
    df = pd.read_csv(RUNS_CSV)
    missing, stats = collect(df)
    if missing:
        for label in missing:
            print(f"{label}  없음")
        return 1
    print_sgd(stats)
    print()
    print_grid(stats)
    print()
    print_adam(stats)
    print()
    return 0 if check(stats) else 1


if __name__ == "__main__":
    raise SystemExit(main())
