"""
새 흐름용 C2 실험 (과적합 대응 / 불균형).

학습·평가는 c1_flow_common을 쓴다. 같은 조건이 c1_flow_runs.csv에 있으면 재사용하고,
새로 학습한 행만 c2_flow_runs.csv에 저장한다. c1_flow_runs.csv에는 쓰지 않는다.

사용법:
  python members/C2/c2_flow_experiment.py --smoke
  python members/C2/c2_flow_experiment.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

C2_DIR = os.path.dirname(os.path.abspath(__file__))
C1_DIR = os.path.abspath(os.path.join(C2_DIR, "..", "C1"))
for cand in (os.environ.get("C1_DIR"), C1_DIR):
    if cand and os.path.isfile(os.path.join(cand, "c1_flow_common.py")):
        C1_DIR = os.path.abspath(cand)
        if C1_DIR not in sys.path:
            sys.path.insert(0, C1_DIR)
        break

import c1_flow_common as flow
from c1_ver2_adam import keras

C2_CSV = os.path.join(C2_DIR, "c2_flow_runs.csv")
SEEDS = list(flow.SEEDS)
NO_EPOCH = "기록 없음"
SMOKE_VAL = 0.688512
C1_CHECK = [
    ("출발", 0.6907),
    ("드롭아웃 0.2", 0.6919),
    ("조기종료 p5", 0.6889),
    ("조기종료 p10", 0.6907),
]
HEADERS = ["후보", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "양성예측", "최고/전체에폭", "실행"]
ALIGNS = ["left"] + ["right"] * 7 + ["left"]
A_HEADERS = HEADERS + ["30에폭 검증PR", "검증 하락", "학습 상승"]
A_ALIGNS = ALIGNS + ["right"] * 3
B_HEADERS = HEADERS[:6] + ["F1"] + HEADERS[6:]
B_ALIGNS = ["left"] + ["right"] * 8 + ["left"]

START = flow.default_cfg(
    hidden=(28, 56, 28), dropout=0.0, drop_where="none",
    optimizer="Adam", lr=0.01, batch=128, max_epochs=30,
    early_stop=0, class_weight="balanced",
)


def cfg_dropout(rate):
    if rate <= 0:
        return flow.default_cfg(**dict(START, dropout=0.0, drop_where="none"))
    return flow.default_cfg(**dict(START, dropout=float(rate), drop_where="all"))


def cfg_es(patience, dropout=0.0):
    base = cfg_dropout(dropout)
    return flow.default_cfg(**dict(base, early_stop=int(patience), max_epochs=100))


EXP_A = [
    ("출발", START),
    ("드롭아웃 0.1", cfg_dropout(0.1)),
    ("드롭아웃 0.2", cfg_dropout(0.2)),
    ("드롭아웃 0.3", cfg_dropout(0.3)),
    ("드롭아웃 0.5", cfg_dropout(0.5)),
    ("조기종료 p5", cfg_es(5)),
    ("조기종료 p10", cfg_es(10)),
    ("드롭아웃 0.2 + 조기종료 p10", cfg_es(10, dropout=0.2)),
]
EXP_B = [
    ("없음", flow.default_cfg(**dict(START, class_weight="none"))),
    ("클래스 가중치", START),
    ("언더샘플링 1:4", flow.default_cfg(**dict(START, class_weight="under_1to4"))),
]


def render(headers, aligns, rows):
    widths = [
        max(flow.disp_width(headers[i]), *(flow.disp_width(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = ["  ".join(flow.pad(headers[i], widths[i], aligns[i]) for i in range(len(headers)))]
    for row in rows:
        lines.append("  ".join(flow.pad(row[i], widths[i], aligns[i]) for i in range(len(row))))
    return "\n".join(lines), widths


def cached_c2(cfg, seed):
    if not os.path.isfile(C2_CSV):
        return None
    df = pd.read_csv(C2_CSV)
    if df.empty:
        return None
    hit = df[(df["cfg"] == flow.key_of(cfg)) & (df["seed"] == seed)]
    return None if hit.empty else hit.sort_values("epoch")


def under_1to4(x, y, seed):
    """c1_ver2_adam.run_one의 under_1to4. 학습 셋만, 시드 고정."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keep = np.concatenate([pos, rng.choice(neg, 4 * len(pos), replace=False)])
    return x[keep], y[keep]


def train_one(cfg, seed, d):
    """flow.run_one과 같되 c2_flow_runs.csv에만 저장하고, under_1to4를 지원한다."""
    cfg = flow.default_cfg(**cfg)
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    x, y = d["X_tr"], d["y_tr"]
    cw = None
    if cfg["class_weight"] == "balanced":
        cw = flow.class_weight_of(y, "balanced")
    elif cfg["class_weight"] == "under_1to4":
        x, y = under_1to4(x, y, seed)
    elif cfg["class_weight"] not in (None, "none"):
        raise ValueError(f"지원하지 않는 class_weight: {cfg['class_weight']}")

    snap = flow.EpochSnapshot()
    callbacks = [snap]
    if cfg["early_stop"]:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=cfg["early_stop"], restore_best_weights=True,
        ))
    else:
        callbacks.append(flow.v2.BestWeights())

    model = flow.build(cfg, d["X_tr"].shape[1])
    t0 = time.perf_counter()
    history = model.fit(
        x, y,
        validation_data=(d["X_va"], d["y_va"]),
        epochs=cfg["max_epochs"],
        batch_size=cfg["batch"],
        class_weight=cw,
        callbacks=callbacks,
        verbose=0,
    ).history
    seconds = round(time.perf_counter() - t0, 1)
    keras_curve = [float(v) for v in history["val_pr_auc"]]
    best_epoch = int(np.argmax(keras_curve)) + 1
    n_epochs = len(keras_curve)
    n_params = int(model.count_params())
    p_va = model.predict(d["X_va"], verbose=0).ravel()
    p_tr = model.predict(d["X_tr"], verbose=0).ravel()
    report = dict(
        report_val_pr=flow._ap(d["y_va"], p_va),
        report_train_pr=flow._ap(d["y_tr"], p_tr),
        **{f"report_{k}": v for k, v in flow._threshold_metrics(d["y_va"], p_va).items()},
    )
    rows = []
    key = flow.key_of(cfg)
    for epoch, (weights, logs) in enumerate(zip(snap.weights, snap.logs), start=1):
        model.set_weights(weights)
        p_tr_e = model.predict(d["X_tr"], verbose=0).ravel()
        p_va_e = model.predict(d["X_va"], verbose=0).ravel()
        m = flow._threshold_metrics(d["y_va"], p_va_e)
        rows.append(dict(
            cfg=key, seed=seed, epoch=epoch,
            val_pr_keras=keras_curve[epoch - 1],
            val_pr_sk=flow._ap(d["y_va"], p_va_e),
            train_pr=flow._ap(d["y_tr"], p_tr_e),
            loss=float(logs["loss"]),
            val_loss=float(logs["val_loss"]),
            recall=m["recall"], precision=m["precision"], f1=m["f1"], pos_rate=m["pos_rate"],
            best_epoch=best_epoch, n_epochs=n_epochs,
            n_params=n_params, seconds=seconds,
            **report,
        ))
    out = pd.DataFrame(rows)[flow.EPOCH_COLS]
    os.makedirs(C2_DIR, exist_ok=True)
    out.to_csv(C2_CSV, mode="a", header=not os.path.isfile(C2_CSV),
               index=False, encoding="utf-8-sig")
    return out


def run_seed(cfg, seed, d, force_train=False):
    cfg = flow.default_cfg(**cfg)
    if not force_train:
        hit = flow.cached_rows(cfg, seed)
        if hit is not None:
            print(f"  [재사용 C1] seed {seed}", flush=True)
            return hit, "재사용"
        hit = cached_c2(cfg, seed)
        if hit is not None:
            print(f"  [재사용 C2] seed {seed}", flush=True)
            return hit, "재사용"
    print(f"  [학습] seed {seed}", flush=True)
    return train_one(cfg, seed, d), "신규"


def run_combo(cfg, d, seeds=None, force_train=False):
    frames, sources = [], []
    for seed in (seeds or SEEDS):
        frame, src = run_seed(cfg, seed, d, force_train=force_train)
        frames.append(frame)
        sources.append(src)
    src = "신규" if "신규" in sources else "재사용"
    return pd.concat(frames, ignore_index=True), src


def overfit_extra(frame):
    drops, rises, e30 = [], [], []
    for seed, sub in frame.groupby("seed"):
        best = int(sub.iloc[0]["best_epoch"])
        at_best = sub[sub["epoch"] == best]
        at_30 = sub[sub["epoch"] == 30]
        if at_best.empty or at_30.empty:
            return [NO_EPOCH, NO_EPOCH, NO_EPOCH]
        best_row, row30 = at_best.iloc[0], at_30.iloc[0]
        e30.append(float(row30["val_pr_sk"]))
        drops.append(float(best_row["val_pr_sk"]) - float(row30["val_pr_sk"]))
        rises.append(float(row30["train_pr"]) - float(best_row["train_pr"]))
    e30 = np.array(e30)
    return [
        f"{float(e30.mean()):.4f}±{float(e30.std(ddof=1)):.4f}",
        f"{float(np.mean(drops)):+.4f}",
        f"{float(np.mean(rises)):+.3f}",
    ]


def combo_cells(label, frame, source, extra=None, with_f1=False):
    reports = flow.seed_reports(frame)
    cells, val_mean, val_std = flow.combo_cells(label, reports)
    cells = cells + [source]
    if with_f1:
        f1 = f"{float(reports['f1'].mean()):.3f}"
        cells = cells[:6] + [f1] + cells[6:]
    if extra:
        cells = cells + extra
    return dict(label=label, cells=cells, val_mean=val_mean, val_std=val_std,
                reports=reports, frame=frame, source=source)


def verdict(start, top):
    diff = top["val_mean"] - start["val_mean"]
    tol = max(top["val_std"], start["val_std"])
    keep = top["label"] == start["label"] or diff <= tol
    return diff, tol, "유지" if keep else "변경"


def print_block(title, condition, headers, aligns, stats, start):
    rows = [item["cells"] for item in stats]
    print(f"=== {title} ===")
    print(condition)
    text, widths = render(headers, aligns, rows)
    print(text)
    print("1위 후보")
    top = max(stats, key=lambda s: s["val_mean"])
    print("  ".join(flow.pad(top["cells"][i], widths[i], aligns[i]) for i in range(len(headers))))
    diff, tol, flag = verdict(start, top)
    print(f"     출발 조건 차이 {diff:+.4f}, 두 표준편차 중 큰 값 {tol:.4f} -> {flag}")


def c1_mean_and_seeds(cfg):
    frames = []
    for seed in SEEDS:
        hit = flow.cached_rows(cfg, seed)
        if hit is None:
            return None, {}
        frames.append(hit)
    df = pd.concat(frames, ignore_index=True)
    reports = flow.seed_reports(df)
    per = {int(r.seed): float(r.val_pr) for r in reports.itertuples()}
    return float(reports["val_pr"].mean()), per


def check_c1(stats):
    print("\n[확인] c1_flow_runs.csv 기존 값과 시드별 비교")
    by_label = {item["label"]: item for item in stats}
    for label, expected in C1_CHECK:
        item = by_label.get(label)
        cfg = dict(next(c for n, c in EXP_A if n == label))
        ref_mean, ref_seeds = c1_mean_and_seeds(cfg)
        print(f"  {label}  기대 평균 {expected:.4f}")
        if item is None:
            print("    이번 결과 없음")
            continue
        got_mean = item["val_mean"]
        print(f"    이번 평균 {got_mean:.4f}  "
              f"{'일치' if round(got_mean, 4) == expected else '불일치'}")
        if ref_mean is None:
            print("    c1_flow_runs.csv에 해당 조건 없음")
            continue
        for seed in SEEDS:
            got = float(item["reports"].loc[item["reports"]["seed"] == seed, "val_pr"].iloc[0])
            ref = ref_seeds.get(seed)
            same = ref is not None and round(got, 6) == round(ref, 6)
            print(f"    seed {seed}  이번 {got:.6f}  c1 {ref:.6f}  {'일치' if same else '불일치'}")


def smoke(d):
    print("[smoke] 출발 조건 시드 42만 학습")
    frame, _src = run_seed(START, 42, d, force_train=True)
    got = float(frame.iloc[0]["report_val_pr"])
    same = round(got, 6) == round(SMOKE_VAL, 6)
    print(f"  이번 {got:.6f}  c1_flow_runs.csv {SMOKE_VAL:.6f}  차이 {got - SMOKE_VAL:+.6f}  "
          f"{'일치' if same else '불일치'}")
    return 0 if same else 1


def main(do_smoke=False):
    t0 = time.perf_counter()
    d = flow.load_data()
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} (test 미사용)")
    print(flow.key_of(START))
    if do_smoke:
        code = smoke(d)
        print(f"전체 {time.perf_counter() - t0:.1f}초")
        return code

    stats_a = []
    for label, cfg in EXP_A:
        print(f"\n[실험 A] {label}", flush=True)
        frame, src = run_combo(cfg, d)
        extra = overfit_extra(frame)
        stats_a.append(combo_cells(label, frame, src, extra=extra))
    print()
    print_block(
        "실험 A 과적합 대응",
        "(28,56,28), Adam 0.01, 배치 128, 클래스 가중치",
        A_HEADERS, A_ALIGNS, stats_a, stats_a[0],
    )
    check_c1(stats_a)

    stats_b = []
    for label, cfg in EXP_B:
        print(f"\n[실험 B] {label}", flush=True)
        frame, src = run_combo(cfg, d)
        stats_b.append(combo_cells(label, frame, src, with_f1=True))
    print()
    print_block(
        "실험 B 불균형 근거",
        "(28,56,28), Adam 0.01, 배치 128, 드롭아웃 0, 조기종료 없음",
        B_HEADERS, B_ALIGNS, stats_b, stats_b[1],
    )
    print(f"\n저장: {C2_CSV}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="출발 조건 시드 42만 학습해 c1_flow_runs.csv 0.688512와 비교")
    raise SystemExit(main(do_smoke=parser.parse_args().smoke))
