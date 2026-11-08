"""
5단계 비교 (C2) · 과적합 대응 기법 한 가지씩

출발은 step04_overfit.compute() 가 돌려준 조건 그대로.
시드 42/43/44, 평가는 flow_core 와 같다(최고 에폭 가중치 → sklearn PR-AUC).

후보 (출발에서 기법 하나만):
  없음, 드롭아웃 0.1/0.2/0.3, 조기종료 p5/p10(최대 100에폭) — flow_runs.csv 재사용
  L2 1e-4/1e-3/1e-2 — 은닉 Dense 에만 kernel L2. flow_core 에 있으면 그 캐시, 없으면 이 파일에서 감싼다
  단순화 (14,28,14)
새로 학습한 결과는 c2_flow_runs.csv 에만 추가한다. flow_runs.csv 와 기존 스크립트는 건드리지 않는다.

사용법: python members/C2/flow/c2_step05b_overfit_compare.py
"""
import inspect
import os
import sys
import time

C1_FLOW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "C1", "flow"))
if C1_FLOW not in sys.path:
    sys.path.insert(0, C1_FLOW)

import flow_core as fc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from step04_overfit import compute as prev  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C2_CSV = os.path.join(HERE, "c2_flow_runs.csv")
FIG_PATH = os.path.join(HERE, "overfit_compare.png")

L2S = [(1e-4, "L2 1e-4"), (1e-3, "L2 1e-3"), (1e-2, "L2 1e-2")]
COLS = ["pr", "train", "gap", "best", "val30", "drop", "recall", "precision"]
# 캐시에 들어 있는 시드별 검증 PR-AUC. 평균은 출발 0.6907, 드롭아웃 0.2 0.6919, 조기종료 p5 0.6889.
EXPECTED_SEED = {
    "없음": {42: 0.6885123181, 43: 0.7013277343, 44: 0.6822383693},
    "드롭아웃 0.2": {42: 0.6818299330, 43: 0.7016876864, 44: 0.6923272723},
    "조기종료 p5": {42: 0.6885123181, 43: 0.6959939647, 44: 0.6822383693},
}
EXPECTED_MEAN = {"없음": 0.6907, "드롭아웃 0.2": 0.6919, "조기종료 p5": 0.6889}

_C2 = None


def core_has_l2():
    src = inspect.getsource(fc.build)
    return "kernel_regularizer" in src and "l2" in src


def build_model(cfg, input_dim):
    """L2 가 flow_core 에 있으면 그대로 쓰고, 없으면 build 를 감싸 은닉층에만 붙인다."""
    l2 = float(cfg.get("l2", 0.0))
    if l2 > 0 and not core_has_l2():
        return _build_hidden_l2(cfg, input_dim, l2)
    return fc.build(cfg, input_dim)


def _build_hidden_l2(cfg, input_dim, l2):
    _, keras, layers = fc.tf_modules()
    hidden = tuple(cfg["hidden"])
    reg = keras.regularizers.l2(l2)
    seq = [layers.Input(shape=(input_dim,))]
    for i, units in enumerate(hidden):
        seq.append(layers.Dense(
            units, activation="relu", kernel_initializer="he_normal",
            kernel_regularizer=reg,
        ))
        is_last = i == len(hidden) - 1
        where = cfg.get("drop_where", "none")
        if cfg["dropout"] > 0 and (where == "all" or (where == "last" and is_last)):
            seq.append(layers.Dropout(cfg["dropout"]))
    seq.append(layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(seq)
    model.compile(
        optimizer=fc.make_optimizer(cfg["optimizer"], cfg["lr"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def _read_c2(refresh=False):
    global _C2
    if _C2 is None or refresh:
        _C2 = pd.read_csv(C2_CSV) if os.path.isfile(C2_CSV) else pd.DataFrame(columns=fc.EPOCH_COLS)
    return _C2


def _c2_hit(cfg, seed):
    df = _read_c2()
    if df.empty:
        return None
    hit = df[(df["cfg"] == fc.key_of(cfg)) & (df["seed"] == seed)]
    return None if hit.empty else hit.sort_values("epoch")


def train_one(cfg, seed):
    """flow_core.run_one 과 같은 절차. 기록은 c2_flow_runs.csv 에만 쓴다."""
    cfg = fc.cfg_of(**cfg)
    hit = _c2_hit(cfg, seed)
    if hit is not None:
        return hit, "재사용"

    _, keras, _ = fc.tf_modules()
    EpochSnapshot, BestWeights = fc._callbacks()
    d = fc.load_data()
    print(f"  [학습] {fc.cfg_line(cfg)} · 시드 {seed}", flush=True)

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    x_fit, y_fit = d["X_tr"], d["y_tr"]
    if cfg["class_weight"] == "under_1to4":
        x_fit, y_fit = fc.under_1to4(x_fit, y_fit, seed)
    cw = fc.class_weight_of(y_fit, cfg["class_weight"])

    snap = EpochSnapshot()
    callbacks = [snap]
    if cfg["early_stop"]:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=cfg["early_stop"], restore_best_weights=True,
        ))
    else:
        callbacks.append(BestWeights())

    model = build_model(cfg, d["X_tr"].shape[1])
    t0 = time.perf_counter()
    history = model.fit(
        x_fit, y_fit,
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
        report_val_pr=fc._ap(d["y_va"], p_va),
        report_train_pr=fc._ap(d["y_tr"], p_tr),
        **{f"report_{k}": v for k, v in fc._threshold_metrics(d["y_va"], p_va).items()},
    )

    rows = []
    k = fc.key_of(cfg)
    for epoch, (weights, logs) in enumerate(zip(snap.weights, snap.logs), start=1):
        model.set_weights(weights)
        p_tr_e = model.predict(d["X_tr"], verbose=0).ravel()
        p_va_e = model.predict(d["X_va"], verbose=0).ravel()
        m = fc._threshold_metrics(d["y_va"], p_va_e)
        rows.append(dict(
            cfg=k, seed=seed, epoch=epoch,
            val_pr_keras=keras_curve[epoch - 1],
            val_pr_sk=fc._ap(d["y_va"], p_va_e),
            train_pr=fc._ap(d["y_tr"], p_tr_e),
            loss=float(logs["loss"]),
            val_loss=float(logs["val_loss"]),
            recall=m["recall"], precision=m["precision"], f1=m["f1"], pos_rate=m["pos_rate"],
            best_epoch=best_epoch, n_epochs=n_epochs,
            n_params=n_params, seconds=seconds,
            **report,
        ))
    out = pd.DataFrame(rows)[fc.EPOCH_COLS]
    out.to_csv(C2_CSV, mode="a", header=not os.path.isfile(C2_CSV), index=False, encoding="utf-8-sig")
    _read_c2(refresh=True)
    return out, "새로 실행"


def rows_of(cfg, allow_train):
    """c2 캐시 → flow_core 캐시 → (허용 시) 새 학습."""
    cfg = fc.cfg_of(**cfg)
    frames, flags = [], []
    for seed in fc.SEEDS:
        hit = _c2_hit(cfg, seed)
        if hit is not None:
            frames.append(hit)
            flags.append("재사용")
            continue
        hit = fc.cached_rows(cfg, seed)
        if hit is not None:
            frames.append(hit)
            flags.append("재사용")
            continue
        if not allow_train:
            raise SystemExit(f"[중단] flow_core 캐시에 없음: {fc.cfg_line(cfg)} 시드 {seed}")
        hit, flag = train_one(cfg, seed)
        frames.append(hit)
        flags.append(flag)
    status = "재사용" if all(x == "재사용" for x in flags) else "새로 실행"
    return pd.concat(frames, ignore_index=True), status


def evaluate(candidates, allow_train):
    results, curves = [], {}
    for label, cfg in candidates:
        rows, status = rows_of(cfg, allow_train)
        results.append(fc.summarize(label, cfg, rows, status))
        curves[label] = rows
    return results, curves


def representatives(results):
    """기법 안에서 PR-AUC 1위."""
    by = {r["label"]: r for r in results}
    groups = [
        ("없음", ["없음"]),
        ("드롭아웃", ["드롭아웃 0.1", "드롭아웃 0.2", "드롭아웃 0.3"]),
        ("L2", ["L2 1e-4", "L2 1e-3", "L2 1e-2"]),
        ("단순화", ["단순화 (14,28,14)"]),
        ("조기종료", ["조기종료 p5", "조기종료 p10"]),
    ]
    reps = []
    for name, labels in groups:
        best = max((by[label] for label in labels), key=lambda r: r["mean"])
        shown = dict(best)
        shown["group"] = name
        shown["src_label"] = best["label"]
        reps.append(shown)
    return reps


def gap_of(r):
    return r["train"] - r["mean"]


def judge(reps, start):
    """PR-AUC 가 출발 표준편차 이내이거나 출발보다 높은 기법 중, 학습-검증 차이가 가장 작은 기법."""
    bound = start["std"]
    floor = start["mean"] - bound
    kept, dropped = [], []
    for r in reps:
        (kept if r["mean"] >= floor else dropped).append(r)
    chosen = min(kept, key=lambda r: (gap_of(r), -r["mean"]))
    passed = ", ".join(f"{r['src_label']} {gap_of(r):+.3f}" for r in kept)
    blocked = ("제외: " + ", ".join(r["src_label"] for r in dropped)) if dropped else "제외된 기법 없음"
    line = (f"판정: 출발 {start['mean']:.4f}±{start['std']:.4f}. "
            f"출발 표준편차 이내 또는 상승: {passed}. {blocked}. "
            f"→ 선택 {chosen['src_label']} (학습-검증 차이 {gap_of(chosen):+.3f})")
    return chosen, line


def check_seeds(results):
    by = {r["label"]: r for r in results}
    print("[확인] 기존 기록과 시드별 비교")
    ok_all = True
    for label, seeds in EXPECTED_SEED.items():
        got = {int(row.seed): float(row.val_pr) for row in by[label]["seeds"].itertuples()}
        for seed, exp in seeds.items():
            ok = abs(got[seed] - exp) < 1e-6
            ok_all = ok_all and ok
            print(f"  {label} 시드 {seed}: 이번 {got[seed]:.4f}  기록 {exp:.4f}  {'일치' if ok else '불일치'}")
        mean = by[label]["mean"]
        exp_m = EXPECTED_MEAN[label]
        ok_m = abs(mean - exp_m) < 5e-5
        ok_all = ok_all and ok_m
        print(f"  {label} 평균: 이번 {mean:.4f}  기록 {exp_m:.4f}  {'일치' if ok_m else '불일치'}")
    if not ok_all:
        raise SystemExit("[중단] 캐시 확인 불일치")


def figure(reps, curves, chosen):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import flow_figures as ff

    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Malgun Gothic" in names:
        plt.rcParams["font.family"] = "Malgun Gothic"
    colors = [ff.INK, ff.BLUE, ff.ORANGE, "#2E7D4F", "#7A4EAB"]
    fig, ax = ff._axes(10, 5.4)
    for r, color in zip(reps, colors):
        df = curves[r["src_label"]]
        n_seed = df["seed"].nunique()
        counts = df.groupby("epoch")["val_loss"].count()
        keep = counts[counts == n_seed].index
        sub = df[df["epoch"].isin(keep)]
        m = sub.groupby("epoch")["val_loss"].mean()
        picked = r["src_label"] == chosen["src_label"]
        ax.plot(m.index, m.values, color=color, lw=3.2 if picked else 2.2, label=r["src_label"])
    ax.set_xlabel("에폭", fontsize=13, color=ff.INK, fontweight="bold")
    ax.set_ylabel("검증 손실", fontsize=13, color=ff.INK, fontweight="bold")
    leg = ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    for t in leg.get_texts():
        t.set_color(ff.INK)
    fig.savefig(FIG_PATH, dpi=200, facecolor=ff.BG, bbox_inches="tight")
    plt.close(fig)
    print(f"그림 저장: {FIG_PATH}")


def main():
    _, start, _ = prev()
    runs_before = os.path.getsize(fc.RUNS_CSV)
    cached = [
        ("없음", start),
        ("드롭아웃 0.1", fc.cfg_of(**dict(start, dropout=0.1))),
        ("드롭아웃 0.2", fc.cfg_of(**dict(start, dropout=0.2))),
        ("드롭아웃 0.3", fc.cfg_of(**dict(start, dropout=0.3))),
        ("조기종료 p5", fc.cfg_of(**dict(start, early_stop=5, max_epochs=100))),
        ("조기종료 p10", fc.cfg_of(**dict(start, early_stop=10, max_epochs=100))),
    ]
    fresh = [(label, fc.cfg_of(**dict(start, l2=v))) for v, label in L2S]
    fresh.append(("단순화 (14,28,14)", fc.cfg_of(**dict(start, hidden=(14, 28, 14)))))

    res_c, curves_c = evaluate(cached, allow_train=False)
    where = "flow_core 캐시" if core_has_l2() else "이 파일에서 은닉층 L2"
    print(f"L2 구현: {where}", flush=True)
    res_f, curves_f = evaluate(fresh, allow_train=True)
    if os.path.getsize(fc.RUNS_CSV) != runs_before:
        raise SystemExit("[중단] flow_runs.csv 가 바뀌었다")

    results = res_c + res_f
    curves = {**curves_c, **curves_f}
    reps = representatives(results)
    start_row = next(r for r in results if r["label"] == "없음")
    chosen, line = judge(reps, start_row)

    fc.header("5단계 비교 · 과적합 대응 기법 (C2)", fc.cfg_line(start))
    print("차이 = 학습PR − 검증PR(최고 에폭 가중치). 검증 하락 = 최고 에폭 검증PR − 마지막 에폭 검증PR.")
    print("조기종료는 30에폭 전에 멈추면 30에폭 검증PR이 비어 있다.")
    print("[전체 후보]")
    print(fc.table(results, COLS))
    print("[기법별 대표 — 기법 안 PR-AUC 1위]")
    print(fc.table(reps, COLS))
    print(line)
    check_seeds(results)
    figure(reps, curves, chosen)
    return chosen


if __name__ == "__main__":
    main()
