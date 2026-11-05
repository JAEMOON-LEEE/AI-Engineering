"""
4단계 · 과적합 확인 (3단계에서 고른 조합)

의심: 3단계 표에서 최고 에폭이 30 중 12 근처 → 이후 18에폭 동안 검증 성능이 오르지 않음.
확인: 최고 에폭 이후 '학습 PR-AUC는 오르고 검증 PR-AUC는 떨어지는가'(검증 하락 > 0, 학습 상승 > 0).
     비교용으로 같은 옵티마이저의 다른 학습률도 함께 출력한다(작은 학습률=덜 배움, 큰 학습률=발산).
그림: figures/step04_val_pr_curve.png      (의심 슬라이드: 검증 PR-AUC, 시드별 정점)
     figures/step04_train_val_pr_curve.png (확인 슬라이드: 학습↑ 검증↓, 차이 표시)
     figures/step04_loss_curve.png         (예비: 검증 손실 반등)

사용법: python step04_overfit.py
"""
import flow_core as fc
from step03_opt_lr import LRS, OPT_TEXT
from step03_opt_lr import compute as prev

EXPECTED = {"Adam 0.01": 0.6907}
COLS = ["pr", "train", "gap", "best", "val30", "drop", "rise"]


def compute():
    _, chosen = prev()
    cands = [(f"{OPT_TEXT[chosen['optimizer']]} {fc.lr_text(lr)}", fc.cfg_of(**dict(chosen, lr=lr))) for lr in LRS]
    res = fc.evaluate(cands)
    me = next(r for r in res if r["cfg"]["lr"] == chosen["lr"])
    overfit = me["val_drop"] > 0 and me["train_rise"] > 0 and me["best_epoch"] < 0.8 * me["n_epochs"]
    return res, chosen, overfit


def figures(chosen):
    import flow_figures as ff
    df = fc.curves(chosen)
    peaks = df.groupby("seed")["best_epoch"].first()
    pk = float(peaks.mean())
    last = int(df["epoch"].max())

    # (1) 검증 PR-AUC: 시드별 정점
    fig, ax = ff._axes()
    for seed, sub in df.groupby("seed"):
        ax.plot(sub["epoch"], sub["val_pr_sk"], lw=1.2, alpha=0.55, label=f"시드 {seed}")
        b = int(sub["best_epoch"].iloc[0])
        v = float(sub.loc[sub["epoch"] == b, "val_pr_sk"].iloc[0])
        ax.plot(b, v, "o", ms=8, color=ff.INK)
        ax.annotate(f"{seed}: {b}에폭", (b, v), textcoords="offset points", xytext=(4, 8), fontsize=10, color=ff.INK)
    m = df.groupby("epoch")["val_pr_sk"].mean()
    ax.plot(m.index, m.values, color=ff.BLUE, lw=3, label="평균")
    ax.axvline(pk, color=ff.MUTED, ls=":", lw=1.5)
    ax.axvspan(pk, last, color="#9AA0A8", alpha=0.12)
    ax.text((pk + last) / 2, ax.get_ylim()[0] + 0.005, "이후 개선 없음", ha="center", fontsize=12, color=ff.MUTED)
    ff.finish(ax, ylabel="검증 PR-AUC")
    ff._save(fig, "step04_val_pr_curve.png")

    # (2) 학습 vs 검증 PR-AUC + 차이
    fig, ax = ff._axes()
    tr = ff.mean_band(ax, df, "train_pr", ff.ORANGE, "학습 PR-AUC", ls="--")
    va = ff.mean_band(ax, df, "val_pr_sk", ff.BLUE, "검증 PR-AUC")
    ax.axvline(pk, color=ff.MUTED, ls=":", lw=1.5)
    ax.text(pk + 0.3, ax.get_ylim()[1], f"최고 에폭 평균 {pk:.1f}", va="top", fontsize=11, color=ff.MUTED,
            bbox=dict(facecolor=ff.BG, edgecolor="none", pad=2))
    gaps = {}
    for e in (round(pk), last):
        g = float(tr[e] - va[e])
        gaps[e] = (float(tr[e]), float(va[e]), g)
        ax.annotate("", xy=(e, tr[e]), xytext=(e, va[e]), arrowprops=dict(arrowstyle="<->", color=ff.INK, lw=1.4))
        ax.text(e + (0.6 if e != last else -0.6), (tr[e] + va[e]) / 2, f"차이 {g:.3f}",
                ha="left" if e != last else "right", fontsize=12, fontweight="bold", color=ff.INK)
    ax.text(last + 0.3, tr[last], f"{tr[last]:.3f}", color=ff.ORANGE, fontsize=11, fontweight="bold", va="center")
    ax.text(last + 0.3, va[last] - 0.012, f"{va[last]:.3f}", color=ff.BLUE, fontsize=11, fontweight="bold", va="top")
    ff.finish(ax)
    ff._save(fig, "step04_train_val_pr_curve.png")

    # (3) 손실
    fig, ax = ff._axes()
    ff.mean_band(ax, df, "loss", ff.ORANGE, "학습 손실")
    vl = ff.mean_band(ax, df, "val_loss", ff.BLUE, "검증 손실")
    e_min = int(vl.idxmin())
    ax.plot(e_min, vl[e_min], "o", ms=9, color=ff.INK)
    ax.annotate(f"최저 {e_min}에폭", (e_min, vl[e_min]), textcoords="offset points", xytext=(6, -16), fontsize=11)
    rise = (vl[last] / vl[e_min] - 1) * 100
    ax.text(last, vl[last], f"  +{rise:.0f}%", fontsize=12, fontweight="bold", color=ff.BLUE, va="center")
    ff.finish(ax, ylabel="손실")
    ff._save(fig, "step04_loss_curve.png")
    return gaps


def main(draw=True):
    res, chosen, overfit = compute()
    fc.header("4단계 · 과적합 확인", fc.cfg_line(chosen, skip=("lr",)))
    print(fc.table(res, COLS))
    print(fc.seed_table(res))
    print("차이 = 학습PR − 검증PR(최고 에폭), 검증 하락 = 최고 에폭 검증PR − 마지막 에폭 검증PR, "
          "학습 상승 = 마지막 에폭 학습PR − 최고 에폭 학습PR")
    print("판정: " + ("과적합 — 최고 에폭 이후 학습은 오르고 검증은 떨어짐. 다음: 드롭아웃·조기종료(C2)."
                     if overfit else "과적합 증상 없음"))
    fc.check_values(res, EXPECTED)
    if draw:
        gaps = figures(chosen)
        for e, (t, v, g) in gaps.items():
            print(f"  {e}에폭: 학습 평균 {t:.4f}, 검증 평균 {v:.4f}, 차이 {g:.3f}")
    return chosen


if __name__ == "__main__":
    main()
