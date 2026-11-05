"""
9단계 그림 · 에폭 확인 (학습 없이 flow_runs.csv만 읽음)

1~30에폭: 시드 42/43/44 선과 시드별 최고 에폭
30~100에폭: 세 시드 평균선 + 최소~최대 띠 → 30 이후로는 떨어지기만 한다
그림: figures/step09_epoch_curve.png

사용법: python step09_epoch_curve.py
"""
import os

os.environ["FLOW_CACHE_ONLY"] = "1"

import flow_core as fc  # noqa: E402
import flow_figures as ff  # noqa: E402
from step08_lr_recheck import compute as prev  # noqa: E402

SEED_COLOR = {42: ff.BLUE, 43: ff.ORANGE, 44: "#2E8B6E"}
MEAN_COLOR = "#31343A"
EXPECTED_BEST = {42: (30, 0.7018), 43: (30, 0.7094), 44: (27, 0.7047)}
EXPECTED_DROP = {30: 0.0048, 50: 0.0184, 100: 0.0334}


def main():
    _, base, _ = prev()
    cfg = fc.cfg_of(**dict(base, max_epochs=100))
    df = fc.curves(cfg)
    mean = df.groupby("epoch")["val_pr_sk"].mean()
    lo = df.groupby("epoch")["val_pr_sk"].min()
    hi = df.groupby("epoch")["val_pr_sk"].max()

    fig, ax = ff._axes(13, 5.6)
    # 1~30: 시드별
    for seed, sub in df.groupby("seed"):
        s = sub[sub["epoch"] <= 30]
        ax.plot(s["epoch"], s["val_pr_sk"], color=SEED_COLOR[seed], lw=2, label=f"시드 {seed}")
    # 30~100: 평균 + 띠
    after = mean.index >= 30
    ax.fill_between(mean.index[after], lo[after], hi[after], color=MEAN_COLOR, alpha=0.12, linewidth=0)
    ax.plot(mean.index[after], mean[after], color=MEAN_COLOR, lw=3, label="30에폭 이후 평균 (띠: 시드 범위)")
    ax.axvline(30, color=ff.MUTED, ls=":", lw=1.6)
    ax.text(65, 0.716, "30에폭 이후 · 떨어지기만 함", ha="center", fontsize=13, color=MEAN_COLOR, fontweight="bold")

    # 시드별 최고 에폭
    LABEL_POS = {43: (25, 0.718), 44: (19, 0.708), 42: (34, 0.664)}
    rows = []
    for seed, sub in df.groupby("seed"):
        b = int(sub["best_epoch"].iloc[0])
        v = float(sub.loc[sub["epoch"] == b, "val_pr_sk"].iloc[0])
        rows.append((seed, b, v, *(float(sub.loc[sub["epoch"] == e, "val_pr_sk"].iloc[0]) for e in (30, 50, 100))))
        ax.plot(b, v, "o", ms=9, color=SEED_COLOR[seed], markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.annotate(f"{seed}: {b}에폭", (b, v), xytext=LABEL_POS[seed], textcoords="data", ha="left" if seed == 42 else "right", va="center",
                    fontsize=11, color=SEED_COLOR[seed], fontweight="bold",
                    bbox=dict(facecolor=ff.BG, edgecolor="none", pad=1.5),
                    arrowprops=dict(arrowstyle="-", color=SEED_COLOR[seed], lw=1.2))
    # 평균 점: 30 / 50 / 100
    for e, dy in ((50, -20), (100, 0)):
        ax.plot(e, mean[e], "o", ms=9, color=MEAN_COLOR, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        if e == 100:
            ax.annotate(f"{mean[e]:.3f}", (e, mean[e]), textcoords="offset points", xytext=(8, 0), va="center",
                        fontsize=12, color=MEAN_COLOR, fontweight="bold")
        else:
            ax.annotate(f"{mean[e]:.3f}", (e, mean[e]),
                        textcoords="offset points", xytext=(8, dy), fontsize=12, color=MEAN_COLOR, fontweight="bold",
                        bbox=dict(facecolor=ff.BG, edgecolor="none", pad=1.5))
    ax.annotate(f"평균 {mean[30]:.3f}", (30, mean[30]), textcoords="offset points", xytext=(10, -22),
                fontsize=12, color=MEAN_COLOR, fontweight="bold", bbox=dict(facecolor=ff.BG, edgecolor="none", pad=1.5))
    ax.set_xlim(1, 104)
    ax.set_ylim(0.60, 0.725)
    ax.set_xticks([1, 10, 20, 30, 50, 70, 100])
    ff.finish(ax, ylabel="검증 PR-AUC", legend=False)
    leg = ax.legend(frameon=False, fontsize=11, loc="lower right", ncol=2)
    for t in leg.get_texts():
        t.set_color(ff.INK)
    ax.set_title("에폭 확인 · 최대 100에폭으로 늘려도 최고는 30에폭 이전", fontsize=15, color=ff.INK,
                 fontweight="bold", pad=12)
    fig.text(0.07, -0.02, "검증 하락(최고 에폭 → 마지막 에폭): 최대 30에폭 +0.005 · 50에폭 +0.018 · 100에폭 +0.033",
             fontsize=11, color=ff.MUTED)
    ff._save(fig, "step09_epoch_curve.png")

    print("시드  최고에폭  그때 검증PR  30에폭  50에폭  100에폭")
    ok = True
    for seed, b, v, v30, v50, v100 in rows:
        print(f"{seed:>4}  {b:>8}  {v:>11.4f}  {v30:.4f}  {v50:.4f}  {v100:.4f}")
        eb, ev = EXPECTED_BEST[seed]
        ok &= (b == eb and round(v, 4) == ev)
    for e, exp in EXPECTED_DROP.items():
        drop = sum(r[2] for r in rows) / 3 - sum(r[{30: 3, 50: 4, 100: 5}[e]] for r in rows) / 3
        print(f"검증 하락 {e}에폭: {drop:+.4f} (기록 {exp:+.4f})")
        ok &= abs(drop - exp) < 5e-5
    print("[확인]", "일치" if ok else "불일치")


if __name__ == "__main__":
    main()
