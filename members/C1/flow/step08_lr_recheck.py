"""
8단계 · 학습률 재확인 (배치·구조를 바꿨으므로)

배치가 바뀌면 업데이트 횟수가, 구조가 바뀌면 한 번에 움직이는 양이 달라져 맞는 학습률도 달라질 수 있다.
후보: 현재 학습률의 1/3, 그대로, 3배 (10배 간격의 로그 중간값 ≈ 3배).
  현재 0.01 이면 → 0.01/3(=0.0033333…, '0.0033' 아님), 0.01, 0.03
선택 규칙: 1위. 이것이 튜닝 마지막 단계(다음: 에폭 확인 → 동결).

사용법: python step08_lr_recheck.py
"""
import flow_core as fc
from step07_structure import compute as prev

EXPECTED = {"학습률 0.01/3": 0.7053, "학습률 0.01": 0.7016, "학습률 0.03": 0.6517}
COLS = ["params", "pr", "train", "gap", "recall", "precision", "best", "val30", "drop", "sec", "status"]


def lr_neighbors(lr0):
    return [lr0 / 3, lr0, round(lr0 * 3, 12)]


def compute():
    _, _, base, _ = prev()
    cands = [(f"학습률 {fc.lr_text(lr)}", fc.cfg_of(**dict(base, lr=lr))) for lr in lr_neighbors(base["lr"])]
    res = fc.evaluate(cands)
    chosen, line = fc.judge(res, f"학습률 {fc.lr_text(base['lr'])}", rule="best")
    return res, chosen["cfg"], line


def main():
    res, chosen, line = compute()
    fc.header("8단계 · 학습률 재확인", fc.cfg_line(chosen, skip=("lr",)))
    print(fc.table(res, COLS))
    print(fc.seed_table(res))
    print(line)
    fc.check_values(res, EXPECTED)
    return chosen


if __name__ == "__main__":
    main()
