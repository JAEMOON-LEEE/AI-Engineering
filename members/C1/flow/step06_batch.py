"""
6단계 · 배치 크기 (5단계까지 정한 설정에서 배치만 변경)

후보 32 / 64 / 128 / 256 (교재 4.5.2 범위). 선택 규칙: 1위.
그림: figures/step06_batch_pr_auc.png (평균 ± 표준편차 오차 막대)

사용법: python step06_batch.py
"""
import flow_core as fc
from _c2_bridge import step05_compute as prev

BATCHES = [32, 64, 128, 256]
EXPECTED = {"배치 32": 0.6891, "배치 64": 0.6945, "배치 128": 0.6919, "배치 256": 0.6891}
COLS = ["pr", "train", "gap", "recall", "precision", "pos", "best", "val30", "drop", "sec", "status"]


def compute():
    _, _, base = prev()
    res = fc.evaluate([(f"배치 {b}", fc.cfg_of(**dict(base, batch=b))) for b in BATCHES])
    cur = f"배치 {base['batch']}"
    chosen, line = fc.judge(res, cur, rule="best")
    return res, chosen["cfg"], line


def main(draw=True):
    res, chosen, line = compute()
    fc.header("6단계 · 배치 크기", fc.cfg_line(chosen, skip=("batch",)))
    print(fc.table(res, COLS))
    print(fc.seed_table(res))
    print(line)
    fc.check_values(res, EXPECTED)
    if draw:
        import flow_figures as ff
        ff.errorbars(res, "step06_batch_pr_auc.png", f"배치 {chosen['batch']}", "배치 크기",
                     sub_labels=[f"{r['seconds']:.1f}초" for r in res])
    return chosen


if __name__ == "__main__":
    main()
