"""
8단계 학습률 재확인 · 학습률 0.01/3 시드별 검증 PR-AUC 곡선.

학습하지 않는다. flow_runs.csv만 읽는다 (flow_core.curves).
기존 3칸 그림 figures/step08_lr_seed_curves.png 는 건드리지 않는다.

사용법: python step08_lr_curve.py
"""
from __future__ import annotations

import os

os.environ["FLOW_CACHE_ONLY"] = "1"

import numpy as np

import flow_core as fc
import flow_figures as ff

LR = 0.01 / 3
SEEDS = (42, 43, 44)
SEED_COLOR = {42: ff.BLUE, 43: ff.ORANGE, 44: "#3D8B74"}
YLIM = (0.60, 0.72)
OUT = "step08_lr_third_seed_curves.png"
EXPECTED_BEST = {42: 30, 43: 30, 44: 27}
EXPECTED_VAL = {42: 0.7018, 43: 0.7094, 44: 0.7047}
# 43은 점 위, 44는 27에폭 점 왼쪽. 42는 점 아래 빈 공간(y≈0.690).
OFFSETS = {
    43: (-10, 12),
    44: (-10, 10),
}
LABEL_BG = dict(facecolor=ff.BG, edgecolor="none", pad=1.8)


def make_cfg():
    return fc.cfg_of(
        hidden=(56, 112, 56),
        dropout=0.2,
        drop_where="all",
        optimizer="Adam",
        lr=LR,
        batch=64,
        max_epochs=30,
        early_stop=0,
        class_weight="balanced",
    )


def collect():
    df = fc.curves(make_cfg())
    reps = []
    for seed in SEEDS:
        sub = df[df["seed"] == seed].sort_values("epoch")
        best = int(sub.iloc[0]["best_epoch"])
        at_best = sub[sub["epoch"] == best].iloc[0]
        at_30 = sub[sub["epoch"] == 30]
        if at_30.empty:
            raise SystemExit(f"30에폭 없음: seed {seed}")
        reps.append(dict(
            seed=seed,
            best_epoch=best,
            best_val=float(at_best["val_pr_sk"]),
            val30=float(at_30.iloc[0]["val_pr_sk"]),
            report=float(sub.iloc[0]["report_val_pr"]),
        ))
    mean = float(np.mean([r["report"] for r in reps]))
    return df, reps, mean


def print_table(reps):
    heads = ["시드", "최고에폭", "그때 검증PR", "30에폭 검증PR"]
    aligns = ["right"] * 4
    body = [
        [str(r["seed"]), str(r["best_epoch"]), f"{r['best_val']:.4f}", f"{r['val30']:.4f}"]
        for r in reps
    ]
    widths = [max(fc.disp_width(row[i]) for row in [heads] + body) for i in range(len(heads))]
    print("\n".join(
        "  ".join(fc.pad(row[i], widths[i], aligns[i]) for i in range(len(row)))
        for row in [heads] + body
    ))


def check(reps):
    print("[확인] 최고 에폭·검증 PR-AUC vs step08_lr_recheck.py")
    ok = True
    for r in reps:
        same_ep = r["best_epoch"] == EXPECTED_BEST[r["seed"]]
        same_pr = round(r["best_val"], 4) == EXPECTED_VAL[r["seed"]]
        ok = ok and same_ep and same_pr
        print(
            f"  seed {r['seed']}  에폭 이번 {r['best_epoch']} 기준 {EXPECTED_BEST[r['seed']]}  "
            f"{'일치' if same_ep else '불일치'}  "
            f"PR 이번 {r['best_val']:.4f} 기준 {EXPECTED_VAL[r['seed']]:.4f}  "
            f"{'일치' if same_pr else '불일치'}"
        )
    print("일치" if ok else "불일치")
    return ok


def draw(df, reps, mean):
    fig, ax = ff._axes(11.4, 5.2)
    for seed in SEEDS:
        sub = df[df["seed"] == seed].sort_values("epoch")
        ax.plot(
            sub["epoch"], sub["val_pr_sk"],
            color=SEED_COLOR[seed], lw=2.2, label=f"시드 {seed}", zorder=3,
        )
    for r in reps:
        ax.plot(
            r["best_epoch"], r["best_val"], "o", ms=8.5,
            color=SEED_COLOR[r["seed"]], markeredgecolor="white",
            markeredgewidth=1.0, zorder=5,
        )
        text = f"{r['seed']}: {r['best_epoch']}에폭 {r['best_val']:.3f}"
        if r["seed"] == 42:
            ax.annotate(
                text, (r["best_epoch"], r["best_val"]),
                xytext=(27.2, 0.690), textcoords="data",
                fontsize=11, color=ff.INK, fontweight="bold",
                ha="center", va="center", bbox=LABEL_BG, zorder=6,
            )
        else:
            dx, dy = OFFSETS[r["seed"]]
            ax.annotate(
                text, (r["best_epoch"], r["best_val"]),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=11, color=ff.INK, fontweight="bold",
                ha="right", va="center", bbox=LABEL_BG, zorder=6,
            )
    ax.axvline(30, color=ff.MUTED, ls=":", lw=1.6, zorder=1)
    ax.set_xlim(1, 30.8)
    ax.set_ylim(*YLIM)
    ax.set_title(
        f"학습률 0.01/3 · 시드별 검증 PR-AUC (평균 {mean:.4f})",
        fontsize=14, color=ff.INK, fontweight="bold", pad=12,
    )
    ax.set_xlabel("에폭", fontsize=13, color=ff.INK, fontweight="bold")
    ax.set_ylabel("검증 PR-AUC", fontsize=13, color=ff.INK, fontweight="bold")
    by = {r["seed"]: r["best_epoch"] for r in reps}
    ax.text(
        0.0, -0.16,
        f"최고 에폭 42: {by[42]} · 43: {by[43]} · 44: {by[44]} — "
        "두 시드가 마지막 에폭에서 아직 오르는 중",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11, color=ff.MUTED, clip_on=False,
    )
    leg = ax.legend(frameon=False, fontsize=12, loc="lower right")
    for t in leg.get_texts():
        t.set_color(ff.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return ff._save(fig, OUT)


def main():
    fc.header(
        "8단계 · 학습률 0.01/3 곡선",
        "(56,112,56), Adam 0.01/3, 배치 64, 최대 30에폭, 드롭아웃 0.2(모든 은닉층), "
        "조기종료 없음, 클래스 가중치",
    )
    df, reps, mean = collect()
    print_table(reps)
    draw(df, reps, mean)
    return 0 if check(reps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
