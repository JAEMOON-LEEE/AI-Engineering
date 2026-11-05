"""
9단계 · 구조가 바뀌었으면 학습률 재확인 (안 바뀌었으면 생략)

모델 크기가 바뀌면 맞는 학습률도 달라질 수 있다(큰 모델은 한 번에 움직이는 양이 큼).
후보와 규칙은 7단계와 같다(1/3, 그대로, 3배 · 1위 선택). 변화가 없으면 반복을 멈춘다.

사용법: python step09_lr_recheck2.py
"""
import flow_core as fc
from step07_lr_recheck import recheck
from step08_structure import compute as prev

EXPECTED = {"학습률 0.01/3": 0.7053, "학습률 0.01": 0.7016, "학습률 0.03": 0.6517}
COLS = ["params", "pr", "train", "gap", "recall", "precision", "best", "val30", "drop", "sec", "status"]


def compute():
    _, base, chosen, _ = prev()
    if chosen["hidden"] == base["hidden"]:
        return None, chosen, "구조가 유지되어 학습률 재확인은 생략"
    return recheck(chosen)


def main():
    res, chosen, line = compute()
    fc.header("9단계 · 구조 변경 후 학습률 재확인", fc.cfg_line(chosen, skip=("lr",)))
    if res is None:
        print(line)
        return chosen
    print(fc.table(res, COLS))
    print(fc.seed_table(res))
    print(line)
    fc.check_values(res, EXPECTED)
    return chosen


if __name__ == "__main__":
    main()
