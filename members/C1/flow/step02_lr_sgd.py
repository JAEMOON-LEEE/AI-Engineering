"""
2단계 · 학습률 그리드 (SGD+NAG 유지, 학습률만 변경)

교재 방식대로 옵티마이저는 논문 값(SGD)을 유지하고 학습률부터 키운다.
후보 0.00005 / 0.001 / 0.01 / 0.1 (10배 간격). 선택 규칙: 1위.

사용법: python step02_lr_sgd.py
"""
import flow_core as fc
from step00_initial import compute as prev  # 1단계(에폭 300)는 설정을 바꾸지 않으므로 0단계에서 이어감

LRS = [5e-5, 0.001, 0.01, 0.1]
EXPECTED = {"lr=0.00005": 0.3432, "lr=0.001": 0.6005, "lr=0.01": 0.6750, "lr=0.1": 0.6783}


def compute():
    _, base = prev()
    cands = [(f"lr={fc.lr_text(lr)}", fc.cfg_of(**dict(base, lr=lr))) for lr in LRS]
    res = fc.evaluate(cands)
    best = fc.pick_best(res)
    return res, best["cfg"]


def main():
    res, chosen = compute()
    fc.header("2단계 · 학습률 그리드 (SGD+NAG)", fc.cfg_line(chosen, skip=("lr",)))
    print(fc.table(res))
    print(fc.seed_table(res))
    print(f"선택: 학습률 {fc.lr_text(chosen['lr'])} (1위). 0.01과 0.1은 표준편차 안이라 사실상 무승부 —"
          " 핵심은 '학습률이 원인이었다'(0.343 → 0.68). 학습률은 다음 단계에서 옵티마이저와 함께 다시 고른다.")
    fc.check_values(res, EXPECTED)
    return chosen


if __name__ == "__main__":
    main()
