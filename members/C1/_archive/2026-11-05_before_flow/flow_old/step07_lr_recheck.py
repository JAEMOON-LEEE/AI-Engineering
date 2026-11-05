"""
7단계 · 학습률 재확인 (배치를 바꿨으므로)

배치가 바뀌면 한 에폭의 업데이트 횟수가 달라져 같은 학습률도 실제 보폭이 달라진다.
후보: 현재 학습률의 1/3, 그대로, 3배 (10배 간격의 로그 중간값 ≈ 3배).
  현재 0.01 이면 → 0.01/3(=0.0033333…, '0.0033' 아님), 0.01, 0.03
선택 규칙: 1위.

사용법: python step07_lr_recheck.py
"""
import flow_core as fc
from step06_batch import compute as prev

EXPECTED = {"학습률 0.01/3": 0.6961, "학습률 0.01": 0.6945, "학습률 0.03": 0.6579}
COLS = ["pr", "train", "gap", "recall", "precision", "best", "val30", "drop", "sec", "status"]


def lr_neighbors(lr0):
    return [lr0 / 3, lr0, round(lr0 * 3, 12)]


def recheck(base):
    cands = [(f"학습률 {fc.lr_text(lr)}", fc.cfg_of(**dict(base, lr=lr))) for lr in lr_neighbors(base["lr"])]
    res = fc.evaluate(cands)
    chosen, line = fc.judge(res, f"학습률 {fc.lr_text(base['lr'])}", rule="best")
    return res, chosen["cfg"], line


def compute():
    _, base, _ = prev()
    return recheck(base)


def main():
    res, chosen, line = compute()
    fc.header("7단계 · 학습률 재확인", fc.cfg_line(chosen, skip=("lr",)))
    print(fc.table(res, COLS))
    print(fc.seed_table(res))
    print(line)
    fc.check_values(res, EXPECTED)
    return chosen


if __name__ == "__main__":
    main()
