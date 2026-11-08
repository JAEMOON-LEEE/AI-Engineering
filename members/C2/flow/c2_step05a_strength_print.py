"""
과적합 대응 · 드롭아웃 비율 / L2 계수 출력. 캐시만 읽고 학습하지 않는다.

출발 조건은 step04_overfit.compute() 결과. 시드 42/43/44.
열과 숫자는 c2_step05b_print.py 와 같다.

사용법: python members/C2/flow/c2_step05a_strength_print.py
"""
import os
import sys

os.environ["FLOW_CACHE_ONLY"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c2_step05b_print as base  # noqa: E402
import flow_core as fc  # noqa: E402

DROPOUT = [0.0, 0.1, 0.2, 0.3, 0.5]
L2S = [0.0, 1e-4, 1e-3, 1e-2]


def _label_drop(rate):
    return "없음(출발)" if rate == 0 else f"드롭아웃 {rate:g}"


def _label_l2(value):
    return "없음(출발)" if value == 0 else f"L2 {value:.0e}".replace("e-0", "e-")


def load(start, specs):
    return [base.summarize(label, base.rows_of(cfg)) for label, cfg in specs]


def judgment(results):
    best = max(results, key=lambda r: r["mean"])
    start = next(r for r in results if r["label"] == "없음(출발)")
    diff = best["mean"] - start["mean"]
    bound = max(best["std"], start["std"])
    return (f"판정: 1위 {best['label']}, 출발과의 차이 {diff:+.4f}, "
            f"두 후보 표준편차 중 큰 값 {bound:.4f} → 대표 {best['label']}")


def section(title, results, start):
    print(f"=== {title} ===")
    print(fc.cfg_line(start, skip=("dropout", "l2", "early_stop")))
    print(base.table(results))
    print(judgment(results))


def main():
    _, start, _ = base.prev()
    drop = [(_label_drop(r), fc.cfg_of(**dict(start, dropout=r))) for r in DROPOUT]
    l2 = [(_label_l2(v), fc.cfg_of(**dict(start, l2=v))) for v in L2S]
    section("드롭아웃 비율 (출발 조건에 드롭아웃만 추가)", load(start, drop), start)
    print()
    section("L2 계수 (출발 조건에 L2만 추가)", load(start, l2), start)


if __name__ == "__main__":
    main()
