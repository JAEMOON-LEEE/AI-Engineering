"""
5단계 비교 그림 · 검증 PR-AUC 곡선

22번 슬라이드(c1 과적합 곡선, overfit_val_pr_curve)와 같은 축.
후보: 없음(출발), 드롭아웃 0.2, L2 1e-4, 단순화 (14,28,14). 조기종료는 그리지 않는다.
에폭별 검증 PR-AUC(val_pr_sk)는 캐시에서 읽고, 시드 42/43/44 평균이다.
기록이 없는 후보만 새로 학습해 c2_step05b_pr_runs.csv 에 둔다.
flow_runs.csv, c2_flow_runs.csv, 기존 그림은 바꾸지 않는다.

사용법: python members/C2/flow/c2_step05b_plot_pr.py
"""
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
PR_CSV = os.path.join(HERE, "c2_step05b_pr_runs.csv")
FIG_PATH = os.path.join(HERE, "overfit_compare_pr_v2.png")
FONT = {
    "bold": os.path.join(C1_FLOW, "fonts", "GothicA1-Bold.ttf"),
    "extrabold": os.path.join(C1_FLOW, "fonts", "GothicA1-ExtraBold.ttf"),
}

# c1_step3_pr_curve.plot_val 과 같은 범위·바탕.
YLIM = (0.58, 0.74)
BG = "#F5F6F7"
TEXT = "#31343A"
LABEL = "#4A4E55"
MUTED = "#6A6E75"
DROP_COLOR = "#2F5BD8"
SEEDS = (42, 43, 44)

# 강조는 드롭아웃만. 나머지는 회색 계열.
STYLE = {
    "없음(출발)": dict(color="#4A4E55", lw=1.35, band=True, band_color="#9AA0A8", z=2),
    "드롭아웃 0.2": dict(color=DROP_COLOR, lw=2.7, band=True, band_color=DROP_COLOR, z=4),
    "L2 1e-4": dict(color="#6B6F76", lw=1.35, band=False, band_color=None, z=3),
    "단순화 (14,28,14)": dict(color="#9AA0A8", lw=1.35, band=False, band_color=None, z=3),
}

_PR = None


def _read_pr(refresh=False):
    global _PR
    if _PR is None or refresh:
        _PR = pd.read_csv(PR_CSV) if os.path.isfile(PR_CSV) else pd.DataFrame(columns=fc.EPOCH_COLS)
    return _PR


def _hit(df, cfg, seed):
    if df is None or df.empty or "cfg" not in df.columns:
        return None
    hit = df[(df["cfg"] == fc.key_of(cfg)) & (df["seed"] == seed)]
    return None if hit.empty else hit.sort_values("epoch")


def _usable(hit):
    if hit is None or "val_pr_sk" not in hit.columns:
        return False
    epochs = set(int(e) for e in hit["epoch"])
    return set(range(1, 31)) <= epochs and hit["val_pr_sk"].notna().all()


def _from_caches(cfg, seed):
    for df in (_read_pr(), _c2(), fc.read_cache()):
        hit = _hit(df, cfg, seed)
        if _usable(hit):
            return hit
    return None


def _c2():
    if not os.path.isfile(C2_CSV):
        return pd.DataFrame(columns=fc.EPOCH_COLS)
    return pd.read_csv(C2_CSV)


def train_one(cfg, seed):
    """기록이 없을 때만. 새 파일 c2_step05b_pr_runs.csv 에만 쓴다."""
    cfg = fc.cfg_of(**cfg)
    hit = _hit(_read_pr(), cfg, seed)
    if _usable(hit):
        return hit

    _, keras, _ = fc.tf_modules()
    EpochSnapshot, BestWeights = fc._callbacks()
    d = fc.load_data()
    print(f"  [학습] {fc.cfg_line(cfg)} · 시드 {seed}", flush=True)
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    x_fit, y_fit = d["X_tr"], d["y_tr"]
    cw = fc.class_weight_of(y_fit, cfg["class_weight"])
    snap = EpochSnapshot()
    model = fc.build(cfg, d["X_tr"].shape[1])
    t0 = time.perf_counter()
    history = model.fit(
        x_fit, y_fit,
        validation_data=(d["X_va"], d["y_va"]),
        epochs=cfg["max_epochs"],
        batch_size=cfg["batch"],
        class_weight=cw,
        callbacks=[snap, BestWeights()],
        verbose=0,
    ).history
    seconds = round(time.perf_counter() - t0, 1)
    keras_curve = [float(v) for v in history["val_pr_auc"]]
    best_epoch = int(np.argmax(keras_curve)) + 1
    n_epochs = len(keras_curve)
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
            loss=float(logs["loss"]), val_loss=float(logs["val_loss"]),
            recall=m["recall"], precision=m["precision"], f1=m["f1"], pos_rate=m["pos_rate"],
            best_epoch=best_epoch, n_epochs=n_epochs,
            n_params=int(model.count_params()), seconds=seconds,
            **report,
        ))
    out = pd.DataFrame(rows)[fc.EPOCH_COLS]
    out.to_csv(PR_CSV, mode="a", header=not os.path.isfile(PR_CSV), index=False, encoding="utf-8-sig")
    _read_pr(refresh=True)
    return out


def load_candidate(label, cfg):
    cfg = fc.cfg_of(**cfg)
    frames, fresh = [], False
    for seed in SEEDS:
        hit = _from_caches(cfg, seed)
        if hit is None:
            hit = train_one(cfg, seed)
            fresh = True
        one = hit[hit["epoch"].between(1, 30)].sort_values("epoch")
        frames.append(one)
    df = pd.concat(frames, ignore_index=True)
    return df, fresh


def mean_best_epoch(df):
    peaks = [int(df.loc[df["seed"] == seed, "best_epoch"].iloc[0]) for seed in SEEDS]
    return float(np.mean(peaks)), peaks


def curve(df):
    epochs = np.arange(1, 31)
    cols = []
    for seed in SEEDS:
        one = df[df["seed"] == seed].sort_values("epoch")
        cols.append(one["val_pr_sk"].to_numpy(dtype=float)[:30])
    arr = np.vstack(cols)
    return epochs, arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0)


def _face(weight, size):
    from matplotlib import font_manager
    prop = font_manager.FontProperties(fname=FONT[weight])
    prop.set_size(size)
    return prop


def plot(series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10.6, 5.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.12, 0.14, 0.82, 0.78])
    ax.set_facecolor(BG)
    ax.set_xlim(1, 30)
    ax.set_ylim(*YLIM)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74])
    tick_face = _face("bold", 12)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(tick_face)
        tick.set_color(TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E3E5EA")
    ax.spines["bottom"].set_color("#E3E5EA")
    ax.tick_params(color="#E3E5EA", length=0)
    label_face = _face("bold", 15)
    note = _face("extrabold", 13)
    ax.set_xlabel("에폭", fontproperties=label_face, color=LABEL)
    ax.set_ylabel("검증 PR-AUC", fontproperties=label_face, color=LABEL, labelpad=8)

    marks = {}
    for label, df in series.items():
        st = STYLE[label]
        epochs, mean, lo, hi = curve(df)
        if st["band"]:
            ax.fill_between(epochs, lo, hi, color=st["band_color"], alpha=0.16, linewidth=0, zorder=1)
        ax.plot(epochs, mean, color=st["color"], lw=st["lw"], label=label, zorder=st["z"], solid_capstyle="round")
        best, peaks = mean_best_epoch(df)
        marks[label] = (best, peaks)
        print(f"  {label}: 시드별 최고 에폭 {peaks} → 평균 {best:.1f}")

    # 22번 그림(c1_step3_pr_curve.plot_both)의 최고 에폭 평균 점선.
    for label, side in (("없음(출발)", "right"), ("드롭아웃 0.2", "left")):
        mark, _ = marks[label]
        ax.axvline(mark, color=MUTED, linestyle=":", linewidth=1.4, zorder=5)
        ax.text(
            mark + (0.35 if side == "right" else -0.35), 0.97,
            f"최고 에폭 평균 {mark:.1f}",
            transform=ax.get_xaxis_transform(),
            fontproperties=note, color=MUTED, ha="left" if side == "right" else "right",
            va="top", zorder=7,
            bbox=dict(boxstyle="square,pad=0.18", facecolor=BG, edgecolor="none"),
        )

    legend = ax.legend(loc="upper left", frameon=False, prop=_face("bold", 13))
    for text in legend.get_texts():
        text.set_color(TEXT)
    fig.savefig(FIG_PATH, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"그림 저장: {FIG_PATH}")


def _size(path):
    return os.path.getsize(path) if os.path.isfile(path) else None


def main():
    _, start, _ = prev()
    watched = [fc.RUNS_CSV, C2_CSV, os.path.join(HERE, "overfit_compare.png")]
    before = {path: _size(path) for path in watched}
    specs = [
        ("없음(출발)", start),
        ("드롭아웃 0.2", fc.cfg_of(**dict(start, dropout=0.2))),
        ("L2 1e-4", fc.cfg_of(**dict(start, l2=1e-4))),
        ("단순화 (14,28,14)", fc.cfg_of(**dict(start, hidden=(14, 28, 14)))),
    ]
    series = {}
    for label, cfg in specs:
        df, fresh = load_candidate(label, cfg)
        series[label] = df
        print(f"{label}: {'새로 기록' if fresh else '캐시'}", flush=True)
    plot(series)
    for path, size in before.items():
        if _size(path) != size:
            raise SystemExit(f"[중단] 기존 파일이 바뀜: {path}")


if __name__ == "__main__":
    main()
