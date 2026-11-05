"""
5단계 (C2 담당) · 과적합 대응 — 드롭아웃 비율 그리드 + 조기종료

출발: C1 3단계에서 고른 조합(4단계에서 과적합 확인). 학습·캐시는 C1/flow/flow_core 를 그대로 쓴다.
후보(최대 30에폭, 최고 에폭 가중치): 드롭아웃 0 / 0.1 / 0.2 / 0.3 / 0.5 → 선택 규칙: 1위
참고(선택 대상 아님): 조기종료 p5 / p10, 드롭아웃 0.2 + 조기종료 p10 (모두 최대 100에폭)
  - 우리는 이미 최고 에폭 가중치로 되돌리므로 조기종료 단독은 같은 모델이 된다(p10 = 출발).
  - 드롭아웃 0.2 + p10 이 조금 높은 것은 최대 에폭이 100이라 더 오래 배운 효과.

사용법: python members/C2/flow/c2_step05_overfit_fix.py
"""
import os
import sys

C1_FLOW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "C1", "flow"))
if C1_FLOW not in sys.path:
    sys.path.insert(0, C1_FLOW)

import flow_core as fc  # noqa: E402
from step04_overfit import compute as prev  # noqa: E402

RATES = [0.0, 0.1, 0.2, 0.3, 0.5]
EXPECTED = {"출발(드롭아웃 0)": 0.6907, "드롭아웃 0.1": 0.6901, "드롭아웃 0.2": 0.6919,
            "드롭아웃 0.3": 0.6783, "드롭아웃 0.5": 0.6533, "조기종료 p5": 0.6889, "조기종료 p10": 0.6907,
            "드롭아웃 0.2 + 조기종료 p10": 0.6946}
COLS = ["pr", "train", "gap", "recall", "precision", "best", "val30", "drop", "rise"]


def compute():
    _, start, _ = prev()
    drop = [("출발(드롭아웃 0)" if r == 0 else f"드롭아웃 {r:g}", fc.cfg_of(**dict(start, dropout=r))) for r in RATES]
    ref = [("조기종료 p5", fc.cfg_of(**dict(start, early_stop=5, max_epochs=100))),
           ("조기종료 p10", fc.cfg_of(**dict(start, early_stop=10, max_epochs=100))),
           ("드롭아웃 0.2 + 조기종료 p10", fc.cfg_of(**dict(start, dropout=0.2, early_stop=10, max_epochs=100)))]
    res_d = fc.evaluate(drop)
    res_r = fc.evaluate(ref)
    return res_d, res_r, fc.pick_best(res_d)["cfg"]


def main():
    res_d, res_r, chosen = compute()
    fc.header("5단계 · 과적합 대응 (C2)", fc.cfg_line(chosen, skip=("dropout",)))
    print("[드롭아웃 비율 — 선택 대상]")
    print(fc.table(res_d, COLS))
    print("[조기종료 — 참고]")
    print(fc.table(res_r, ["pr", "train", "gap", "best"]))
    print(fc.seed_table(res_d))
    best = fc.pick_best(res_d)
    print(f"선택: {best['label']} (드롭아웃 후보 중 1위). 드롭아웃은 최고 에폭을 늦추고(과적합 완화) "
          "0.3부터는 성능이 떨어짐. 조기종료는 최고 에폭 가중치를 이미 쓰고 있어 추가 효과 없음 → 미채택.")
    fc.check_values(res_d + res_r, EXPECTED)
    return chosen


if __name__ == "__main__":
    main()
