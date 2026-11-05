"""
0단계 · 초기값으로 첫 실행

초기값 근거
  구조 28-56-28, ReLU, SGD 학습률 0.00005 ... Tanvir et al. 2023 (논문 3.7절)
  SGD 모멘텀 0.9 + 네스테로프 ............... 논문은 sklearn 구현 → sklearn 기본값
  배치 128, 최대 30에폭 ...................... 논문에 없음 → 교재 5.4.1
  he_normal 초기화 ........................... 교재 5.5.4 (ReLU에는 He 초기화)
  클래스 가중치 balanced ..................... 논문과 다른 유일한 설정. 구매 15.6% 불균형을 학습 전에 알고 있었고
                                              팀 합의(9/28, 재현율 중심), 기준선(로지스틱·RF)도 balanced
  드롭아웃·조기종료 없음 ..................... 논문에 없음

사용법: python step00_initial.py
"""
import flow_core as fc

INITIAL = fc.cfg_of(hidden=(28, 56, 28), optimizer="SGD", lr=5e-5, batch=128, max_epochs=30,
                    dropout=0.0, early_stop=0, class_weight="balanced")
EXPECTED = {"초기값": 0.3432}


def compute():
    res = fc.evaluate([("초기값", INITIAL)])
    return res, INITIAL


def main():
    res, cfg = compute()
    r = res[0]
    fc.header("0단계 · 초기값 첫 실행", fc.cfg_line(cfg))
    print(fc.table(res))
    print(fc.seed_table(res))
    late = r["best_epoch"] >= 0.8 * r["n_epochs"]
    print(f"증상: 최고 에폭 {round(r['best_epoch'])}/{round(r['n_epochs'])}"
          + (" → 마지막까지 검증 성능이 오름(덜 배움). 다음: 에폭을 늘려 본다." if late else ""))
    fc.check_values(res, EXPECTED)
    return cfg


if __name__ == "__main__":
    main()
