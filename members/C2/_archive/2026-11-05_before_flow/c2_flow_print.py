"""
c1_flow_runs.csv와 c2_flow_runs.csv만 읽어 실험 A·B 표를 출력한다.
학습하지 않는다.

사용법:
  python members/C2/c2_flow_print.py
"""

from __future__ import annotations

import json
import os
import unicodedata

import numpy as np
import pandas as pd

C2_DIR = os.path.dirname(os.path.abspath(__file__))
C1_CSV = os.path.abspath(os.path.join(C2_DIR, "..", "C1", "c1_flow_runs.csv"))
C2_CSV = os.path.join(C2_DIR, "c2_flow_runs.csv")
SEEDS = (42, 43, 44)
NO = "기록 없음"

A_HEADERS = [
    "후보", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "양성예측",
    "최고/전체에폭", "실제 학습 에폭 수", "최대 에폭",
    "30에폭 검증PR", "마지막 에폭 검증PR", "검증 하락", "학습 상승",
]
A_ALIGNS = ["left"] + ["right"] * 13
A_SEED_HEADERS = [
    "후보", "시드", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "양성예측",
    "최고에폭", "실제에폭", "최대에폭",
    "30에폭 검증PR", "마지막 에폭 검증PR", "검증 하락", "학습 상승",
]
A_SEED_ALIGNS = ["left"] + ["right"] * 14
B_HEADERS = [
    "후보", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "F1", "양성예측", "최고/전체에폭",
]
B_ALIGNS = ["left"] + ["right"] * 8
B_SEED_HEADERS = [
    "후보", "시드", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "F1", "양성예측",
    "최고에폭", "실제에폭",
]
B_SEED_ALIGNS = ["left"] + ["right"] * 10


def default_cfg(**updates):
    cfg = dict(
        hidden=(28, 56, 28), dropout=0.0, drop_where="none",
        optimizer="Adam", lr=0.01, batch=128, max_epochs=30,
        early_stop=0, class_weight="balanced",
    )
    cfg.update(updates)
    if not isinstance(cfg["hidden"], tuple):
        cfg["hidden"] = tuple(cfg["hidden"])
    cfg["dropout"] = float(cfg["dropout"])
    cfg["lr"] = float(cfg["lr"])
    cfg["batch"] = int(cfg["batch"])
    cfg["max_epochs"] = int(cfg["max_epochs"])
    cfg["early_stop"] = int(cfg["early_stop"])
    return cfg


def key_of(cfg):
    payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(cfg.items())}
    return json.dumps(payload)


def cfg_dropout(rate):
    if rate <= 0:
        return default_cfg()
    return default_cfg(dropout=float(rate), drop_where="all")


def cfg_es(patience, dropout=0.0):
    base = cfg_dropout(dropout)
    return default_cfg(**dict(base, early_stop=int(patience), max_epochs=100))


EXP_A = [
    ("출발", default_cfg()),
    ("드롭아웃 0.1", cfg_dropout(0.1)),
    ("드롭아웃 0.2", cfg_dropout(0.2)),
    ("드롭아웃 0.3", cfg_dropout(0.3)),
    ("드롭아웃 0.5", cfg_dropout(0.5)),
    ("조기종료 p5", cfg_es(5)),
    ("조기종료 p10", cfg_es(10)),
    ("드롭아웃 0.2 + 조기종료 p10", cfg_es(10, dropout=0.2)),
]
EXP_B = [
    ("없음", default_cfg(class_weight="none")),
    ("클래스 가중치", default_cfg()),
    ("언더샘플링 1:4", default_cfg(class_weight="under_1to4")),
]


def disp_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def pad(text, width, align):
    gap = width - disp_width(text)
    if gap <= 0:
        return text
    return (" " * gap + text) if align == "right" else (text + " " * gap)


def render(headers, aligns, rows):
    widths = [
        max(disp_width(headers[i]), *(disp_width(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = ["  ".join(pad(headers[i], widths[i], aligns[i]) for i in range(len(headers)))]
    for row in rows:
        lines.append("  ".join(pad(row[i], widths[i], aligns[i]) for i in range(len(row))))
    return "\n".join(lines)


def load_all():
    frames = []
    for path, src in ((C1_CSV, "C1"), (C2_CSV, "C2")):
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        df["_src"] = src
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_combo(df, cfg):
    key = key_of(cfg)
    hit = df[df["cfg"] == key]
    parts = []
    for seed in SEEDS:
        one = hit[hit["seed"] == seed]
        if one.empty:
            return None
        if (one["_src"] == "C1").any():
            one = one[one["_src"] == "C1"]
        parts.append(one.sort_values("epoch"))
    return pd.concat(parts, ignore_index=True)


def seed_report(sub):
    r = sub.iloc[0]
    return dict(
        seed=int(r["seed"]),
        val_pr=float(r["report_val_pr"]),
        train_pr=float(r["report_train_pr"]),
        recall=float(r["report_recall"]),
        precision=float(r["report_precision"]),
        f1=float(r["report_f1"]),
        pos_rate=float(r["report_pos_rate"]),
        best_epoch=int(r["best_epoch"]),
        n_epochs=int(r["n_epochs"]),
    )


def at_epoch(sub, epoch, col):
    hit = sub[sub["epoch"] == epoch]
    if hit.empty:
        return None
    return float(hit.iloc[0][col])


def fmt_pm(values, digits):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return NO
    if digits == 4:
        return f"{float(arr.mean()):.4f}±{float(arr.std(ddof=1)):.4f}" if arr.size > 1 else f"{float(arr[0]):.4f}"
    return f"{float(arr.mean()):.3f}"


def fmt_signed(values, digits):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return NO
    mean = float(arr.mean())
    return f"{mean:+.{digits}f}"


def epoch_stats(frame):
    e30_val, last_val, drop, rise = [], [], [], []
    for seed in SEEDS:
        sub = frame[frame["seed"] == seed].sort_values("epoch")
        rep = seed_report(sub)
        best = rep["best_epoch"]
        last = rep["n_epochs"]
        v30 = at_epoch(sub, 30, "val_pr_sk")
        t30 = at_epoch(sub, 30, "train_pr")
        v_best = at_epoch(sub, best, "val_pr_sk")
        t_best = at_epoch(sub, best, "train_pr")
        v_last = at_epoch(sub, last, "val_pr_sk")
        e30_val.append(v30)
        last_val.append(v_last)
        if v30 is None or v_best is None:
            drop.append(None)
        else:
            drop.append(v_best - v30)
        if t30 is None or t_best is None:
            rise.append(None)
        else:
            rise.append(t30 - t_best)
    return dict(
        e30=fmt_pm(e30_val, 4),
        last=fmt_pm(last_val, 4),
        drop=fmt_signed(drop, 4),
        rise=fmt_signed(rise, 3),
        e30_raw=e30_val,
        last_raw=last_val,
        drop_raw=drop,
        rise_raw=rise,
    )


def summarize_a(label, cfg, frame):
    reps = [seed_report(frame[frame["seed"] == seed]) for seed in SEEDS]
    val = np.array([r["val_pr"] for r in reps])
    train = np.array([r["train_pr"] for r in reps])
    val_mean, train_mean = float(val.mean()), float(train.mean())
    val_std = float(val.std(ddof=1))
    actual = ",".join(str(r["n_epochs"]) for r in reps)
    ep = epoch_stats(frame)
    cells = [
        label,
        f"{val_mean:.4f}±{val_std:.4f}",
        f"{train_mean:.3f}",
        f"{train_mean - val_mean:+.3f}",
        f"{float(np.mean([r['recall'] for r in reps])):.3f}",
        f"{float(np.mean([r['precision'] for r in reps])):.4f}",
        f"{float(np.mean([r['pos_rate'] for r in reps])):.3f}",
        f"{int(round(float(np.mean([r['best_epoch'] for r in reps]))))}/"
        f"{int(round(float(np.mean([r['n_epochs'] for r in reps]))))}",
        actual,
        str(int(cfg["max_epochs"])),
        ep["e30"], ep["last"], ep["drop"], ep["rise"],
    ]
    seed_rows = []
    for r, v30, vlast, d, u in zip(reps, ep["e30_raw"], ep["last_raw"], ep["drop_raw"], ep["rise_raw"]):
        seed_rows.append([
            label, str(r["seed"]),
            f"{r['val_pr']:.4f}", f"{r['train_pr']:.3f}", f"{r['train_pr'] - r['val_pr']:+.3f}",
            f"{r['recall']:.3f}", f"{r['precision']:.4f}", f"{r['pos_rate']:.3f}",
            str(r["best_epoch"]), str(r["n_epochs"]), str(int(cfg["max_epochs"])),
            NO if v30 is None else f"{v30:.4f}",
            NO if vlast is None else f"{vlast:.4f}",
            NO if d is None else f"{d:+.4f}",
            NO if u is None else f"{u:+.3f}",
        ])
    return dict(label=label, cells=cells, seed_rows=seed_rows, val_mean=val_mean, val_std=val_std)


def summarize_b(label, frame):
    reps = [seed_report(frame[frame["seed"] == seed]) for seed in SEEDS]
    val = np.array([r["val_pr"] for r in reps])
    train = np.array([r["train_pr"] for r in reps])
    val_mean, train_mean = float(val.mean()), float(train.mean())
    val_std = float(val.std(ddof=1))
    cells = [
        label,
        f"{val_mean:.4f}±{val_std:.4f}",
        f"{train_mean:.3f}",
        f"{train_mean - val_mean:+.3f}",
        f"{float(np.mean([r['recall'] for r in reps])):.3f}",
        f"{float(np.mean([r['precision'] for r in reps])):.4f}",
        f"{float(np.mean([r['f1'] for r in reps])):.3f}",
        f"{float(np.mean([r['pos_rate'] for r in reps])):.3f}",
        f"{int(round(float(np.mean([r['best_epoch'] for r in reps]))))}/"
        f"{int(round(float(np.mean([r['n_epochs'] for r in reps]))))}",
    ]
    seed_rows = []
    for r in reps:
        seed_rows.append([
            label, str(r["seed"]),
            f"{r['val_pr']:.4f}", f"{r['train_pr']:.3f}", f"{r['train_pr'] - r['val_pr']:+.3f}",
            f"{r['recall']:.3f}", f"{r['precision']:.4f}", f"{r['f1']:.3f}", f"{r['pos_rate']:.3f}",
            str(r["best_epoch"]), str(r["n_epochs"]),
        ])
    return dict(label=label, cells=cells, seed_rows=seed_rows, val_mean=val_mean, val_std=val_std)


def collect(df, specs, kind):
    stats, missing = [], []
    for label, cfg in specs:
        frame = load_combo(df, cfg)
        if frame is None:
            missing.append(label)
            continue
        stats.append(summarize_a(label, cfg, frame) if kind == "A" else summarize_b(label, frame))
    return stats, missing


def print_missing(missing):
    for label in missing:
        print(f"{label}  없음")


def print_verdict(stats, start_label):
    start = next(s for s in stats if s["label"] == start_label)
    top = max(stats, key=lambda s: s["val_mean"])
    diff = top["val_mean"] - start["val_mean"]
    tol = max(top["val_std"], start["val_std"])
    keep = top["label"] == start["label"] or diff <= tol
    if keep:
        print(f"판정: 1위 {top['label']}. 출발({start_label})과의 차이 {diff:+.4f}, "
              f"두 후보 표준편차 중 큰 값 {tol:.4f} -> 유지({start_label})")
    else:
        print(f"판정: 1위 {top['label']}. 출발({start_label})과의 차이 {diff:+.4f}, "
              f"두 후보 표준편차 중 큰 값 {tol:.4f} -> 변경({top['label']})")


def main():
    df = load_all()
    if df.empty:
        print("저장된 결과가 없습니다.")
        return 1

    stats_a, miss_a = collect(df, EXP_A, "A")
    print("=== 실험 A 과적합 대응 ===")
    print("(28,56,28), Adam 0.01, 배치 128, 클래스 가중치, 시드 42/43/44")
    if miss_a:
        print_missing(miss_a)
    if stats_a:
        print(render(A_HEADERS, A_ALIGNS, [s["cells"] for s in stats_a]))
        print()
        print("시드별")
        seed_rows = [row for s in stats_a for row in s["seed_rows"]]
        print(render(A_SEED_HEADERS, A_SEED_ALIGNS, seed_rows))
        print()
        print_verdict(stats_a, "출발")

    print()
    stats_b, miss_b = collect(df, EXP_B, "B")
    print("=== 실험 B 불균형 근거 ===")
    print("(28,56,28), Adam 0.01, 배치 128, 드롭아웃 0, 조기종료 없음, 최대 30에폭")
    if miss_b:
        print_missing(miss_b)
    if stats_b:
        print(render(B_HEADERS, B_ALIGNS, [s["cells"] for s in stats_b]))
        print()
        print("시드별")
        seed_rows = [row for s in stats_b for row in s["seed_rows"]]
        print(render(B_SEED_HEADERS, B_SEED_ALIGNS, seed_rows))
        print()
        print_verdict(stats_b, "클래스 가중치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
