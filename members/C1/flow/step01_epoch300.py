"""
1단계 · 에폭만 늘려 보기 (30 → 300)

0단계에서 최고 에폭이 30/30(끝까지 오름)이라 '에폭이 부족한가?'를 확인한다.
나머지 설정은 0단계 그대로. 300에폭은 처음 한 번 학습에 시간이 걸린다(이후 캐시 재사용).

사용법: python step01_epoch300.py
"""
import flow_core as fc
from step00_initial import compute as prev

EXPECTED = {"30에폭": 0.3432}


def compute():
    _, base = prev()
    cands = [("30에폭", base), ("300에폭", fc.cfg_of(**dict(base, max_epochs=300)))]
    res = fc.evaluate(cands)
    return res, base   # 결론: 에폭이 원인이 아니므로 설정은 그대로 다음 단계로


def main():
    res, base = compute()
    fc.header("1단계 · 에폭 늘리기", fc.cfg_line(base, skip=("max_epochs",)))
    print(fc.table(res))
    print(fc.seed_table(res))
    r300 = res[1]
    print(f"결론: 300에폭도 최고 에폭 {round(r300['best_epoch'])}/{round(r300['n_epochs'])} — 계속 오르기만 하고 "
          f"PR-AUC {r300['mean']:.4f}. 에폭을 10배 늘려도 느리다 → 학습률(보폭)을 의심한다.")
    fc.check_values(res, EXPECTED)
    return base


if __name__ == "__main__":
    main()
