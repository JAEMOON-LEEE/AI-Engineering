"""
3단계 · 옵티마이저 × 학습률 그리드

SGD+NAG / RMSprop / Adam × 0.00005 / 0.001 / 0.01 / 0.1 = 12조합 × 시드 3개.
선택 규칙: 12조합 중 1위. 히트맵 figures/step03_opt_lr_heatmap.png

사용법: python step03_opt_lr.py
"""
import flow_core as fc
from step02_lr_sgd import compute as prev

OPTS = ["SGD", "RMSprop", "Adam"]
OPT_TEXT = {"SGD": "SGD+NAG", "RMSprop": "RMSprop", "Adam": "Adam"}
LRS = [5e-5, 0.001, 0.01, 0.1]
EXPECTED = {"Adam 0.01": 0.6907, "RMSprop 0.01": 0.6827, "SGD+NAG 0.1": 0.6783}


def compute():
    _, base = prev()
    cands = [(f"{OPT_TEXT[o]} {fc.lr_text(lr)}", fc.cfg_of(**dict(base, optimizer=o, lr=lr)))
             for o in OPTS for lr in LRS]
    res = fc.evaluate(cands)
    return res, fc.pick_best(res)["cfg"]


def figure(res, chosen):
    import flow_figures as ff
    vals = [[next(r["mean"] for r in res if r["cfg"]["optimizer"] == o and r["cfg"]["lr"] == lr)
             for lr in LRS] for o in OPTS]
    sel = (OPTS.index(chosen["optimizer"]), LRS.index(chosen["lr"]))
    ff.heatmap(vals, [OPT_TEXT[o] for o in OPTS], [fc.lr_text(lr) for lr in LRS],
               "step03_opt_lr_heatmap.png", selected=sel, xlabel="학습률")


def main(draw=True):
    res, chosen = compute()
    fc.header("3단계 · 옵티마이저 × 학습률 그리드", fc.cfg_line(chosen, skip=("optimizer", "lr")))
    print(fc.table(res))
    print("옵티마이저별 1위 학습률")
    print(fc.table([max((r for r in res if r["cfg"]["optimizer"] == o), key=lambda r: r["mean"]) for o in OPTS]))
    best = fc.pick_best(res)
    print(f"선택: {best['label']} (1위 선택 규칙)")
    print(f"관찰: 선택된 조합의 최고 에폭 {round(best['best_epoch'])}/{round(best['n_epochs'])} — "
          "30에폭 한참 전에 정점 → 과적합 의심(4단계에서 확인)")
    fc.check_values(res, EXPECTED)
    if draw:
        figure(res, chosen)
    return chosen


if __name__ == "__main__":
    main()
