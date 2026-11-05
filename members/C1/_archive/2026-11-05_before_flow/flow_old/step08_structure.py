"""
8단계 · 구조 확인 (7단계까지 정한 학습 조건에서 은닉층 구조만 변경)

논문 구조 28-56-28 을 기준으로 깊이(1~3층) × 폭(×0.5 / ×1 / ×2) 9개.
선택 규칙(확인 단계, 실행 전에 정함): 1위와 현재 구조의 차이가 두 후보 표준편차 중 큰 값 이내면 현재 구조 유지.
그림: figures/step08_structure_heatmap.png

사용법: python step08_structure.py
"""
import flow_core as fc
from step07_lr_recheck import compute as prev

DEPTHS = {1: [(14,), (28,), (56,)], 2: [(14, 28), (28, 56), (56, 112)], 3: [(14, 28, 14), (28, 56, 28), (56, 112, 56)]}
WIDTHS = ["×0.5", "×1", "×2"]
EXPECTED = {"3층 (28,56,28)": 0.6961, "3층 (56,112,56)": 0.7053}
COLS = ["params", "pr", "train", "gap", "recall", "precision", "best", "val30", "drop", "sec", "status"]


def label(h):
    return f"{len(h)}층 ({','.join(str(u) for u in h)})"


def compute():
    _, base, _ = prev()
    cands = [(label(h), fc.cfg_of(**dict(base, hidden=h))) for d in DEPTHS for h in DEPTHS[d]]
    res = fc.evaluate(cands)
    chosen, line = fc.judge(res, label(base["hidden"]), rule="keep")
    return res, base, chosen["cfg"], line


def main(draw=True):
    res, base, chosen, line = compute()
    fc.header("8단계 · 구조 확인", fc.cfg_line(base, skip=("hidden",)))
    print(fc.table(res, COLS))
    print(fc.seed_table(res))
    print(line)
    fc.check_values(res, EXPECTED)
    if draw:
        import flow_figures as ff
        get = {r["cfg"]["hidden"]: r for r in res}
        vals = [[get[h]["mean"] for h in DEPTHS[d]] for d in DEPTHS]
        sub = [[f"가중치 {get[h]['n_params']:,}" for h in DEPTHS[d]] for d in DEPTHS]
        pos = {h: (i, j) for i, d in enumerate(DEPTHS) for j, h in enumerate(DEPTHS[d])}
        best = fc.pick_best(res)["cfg"]["hidden"]
        ff.heatmap(vals, [f"{d}층" for d in DEPTHS], WIDTHS, "step08_structure_heatmap.png",
                   selected=pos[chosen["hidden"]], best=pos[best], xlabel="폭 (논문 구조 대비)", sub=sub)
    return chosen


if __name__ == "__main__":
    main()
