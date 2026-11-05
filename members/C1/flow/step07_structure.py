"""
7단계 · 구조 (6단계까지 정한 설정에서 은닉층 구조만 변경)

논문 구조 28-56-28 을 기준으로 깊이(1~3층) × 폭(×0.5 / ×1 / ×2) 9개.
선택 규칙: 1위 (옵티마이저·학습률·드롭아웃·배치와 같은 규칙으로 통일).
그림: figures/step07_structure_heatmap.png

사용법: python step07_structure.py
"""
import flow_core as fc
from step06_batch import compute as prev

DEPTHS = {1: [(14,), (28,), (56,)], 2: [(14, 28), (28, 56), (56, 112)], 3: [(14, 28, 14), (28, 56, 28), (56, 112, 56)]}
WIDTHS = ["×0.5", "×1", "×2"]
EXPECTED = {"3층 (28,56,28)": 0.6945, "3층 (56,112,56)": 0.7016, "1층 (14)": 0.6633, "2층 (28,56)": 0.6938}
COLS = ["params", "pr", "train", "gap", "recall", "precision", "best", "val30", "drop", "sec", "status"]


def label(h):
    return f"{len(h)}층 ({','.join(str(u) for u in h)})"


def compute():
    _, base, _ = prev()
    cands = [(label(h), fc.cfg_of(**dict(base, hidden=h))) for d in DEPTHS for h in DEPTHS[d]]
    res = fc.evaluate(cands)
    chosen, line = fc.judge(res, label(base["hidden"]), rule="best")
    return res, base, chosen["cfg"], line


def main(draw=True):
    res, base, chosen, line = compute()
    fc.header("7단계 · 구조", fc.cfg_line(base, skip=("hidden",)))
    print(fc.table(res, COLS))
    print("층별 1위")
    print(fc.table([max((r for r in res if len(r["cfg"]["hidden"]) == d), key=lambda r: r["mean"]) for d in DEPTHS],
                   COLS))
    print(fc.seed_table(res))
    print(line)
    fc.check_values(res, EXPECTED)
    if draw:
        import flow_figures as ff
        get = {r["cfg"]["hidden"]: r for r in res}
        vals = [[get[h]["mean"] for h in DEPTHS[d]] for d in DEPTHS]
        sub = [[f"가중치 {get[h]['n_params']:,}" for h in DEPTHS[d]] for d in DEPTHS]
        pos = {h: (i, j) for i, d in enumerate(DEPTHS) for j, h in enumerate(DEPTHS[d])}
        ff.heatmap(vals, [f"{d}층" for d in DEPTHS], WIDTHS, "step07_structure_heatmap.png",
                   selected=pos[chosen["hidden"]], xlabel="폭 (논문 구조 대비)", sub=sub)
    return chosen


if __name__ == "__main__":
    main()
