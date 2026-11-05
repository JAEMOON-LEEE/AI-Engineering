"""
(C2 담당, 질문 대비) 불균형 처리 확인 — 없음 / 클래스 가중치 / 언더샘플링 1:4

클래스 가중치는 0단계부터 초기값으로 넣었다(논문과 다른 유일한 설정).
여기서는 3단계 조합(학습이 잘 되는 상태)에서 '가중치를 넣어도 손해가 없었는가'를 확인한다.
모델 비교는 임계값과 무관한 PR-AUC로 하고, 재현율·정밀도·F1(임계값 0.5)은 참고값이다.
최종 임계값은 최종 모델을 정한 뒤 검증 셋으로 따로 정한다.

사용법: python members/C2/flow/c2_imbalance_check.py
"""
import os
import sys

C1_FLOW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "C1", "flow"))
if C1_FLOW not in sys.path:
    sys.path.insert(0, C1_FLOW)

import flow_core as fc  # noqa: E402
from step03_opt_lr import compute as prev  # noqa: E402

EXPECTED = {"없음": 0.6834, "클래스 가중치": 0.6907, "언더샘플링 1:4": 0.6772}


def main():
    _, start = prev()
    cands = [("없음", fc.cfg_of(**dict(start, class_weight="none"))),
             ("클래스 가중치", start),
             ("언더샘플링 1:4", fc.cfg_of(**dict(start, class_weight="under_1to4")))]
    res = fc.evaluate(cands)
    fc.header("불균형 처리 확인 (C2, 참고)", fc.cfg_line(start, skip=("class_weight",)))
    print(fc.table(res, ["pr", "train", "gap", "recall", "precision", "f1", "pos", "best"]))
    print(fc.seed_table(res))
    print("읽는 법: PR-AUC는 가중치 유무가 시드 흔들림 수준. 가중치는 임계값 0.5에서 구매 쪽으로 확률을 올려 "
          "재현율을 높임. 넣어도 성능 손해는 없었다.")
    fc.check_values(res, EXPECTED)


if __name__ == "__main__":
    main()
