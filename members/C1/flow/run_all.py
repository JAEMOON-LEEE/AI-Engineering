"""
처음부터 끝까지 순서대로 실행(0 → 8단계, 5단계는 C2). 결과는 flow_runs.csv 를 재사용하고 없는 것만 학습한다.

  python run_all.py              전체 실행 + 그림
  python run_all.py --no-fig     그림 없이 표만
  set FLOW_CACHE_ONLY=1 (Windows) / FLOW_CACHE_ONLY=1 (bash)  → 학습하지 않고 캐시만 사용

마지막에 최종 설정(아직 에폭 확인·동결 전)을 출력한다.
"""
import argparse

import flow_core as fc
import step00_initial
import step01_epoch300
import step02_lr_sgd
import step03_opt_lr
import step04_overfit
import step06_batch
import step07_structure
import step08_lr_recheck
from _c2_bridge import C2_FLOW  # noqa: F401  (경로 등록)
import c2_step05_overfit_fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    draw = not a.no_fig
    step00_initial.main()
    step01_epoch300.main()
    step02_lr_sgd.main()
    step03_opt_lr.main(draw=draw)
    step04_overfit.main(draw=draw)
    c2_step05_overfit_fix.main()
    step06_batch.main(draw=draw)
    step07_structure.main(draw=draw)
    final = step08_lr_recheck.main()
    print("\n=== 최종 설정 (에폭 확인·동결 전) ===")
    print(fc.cfg_line(final))
    print(f"학습률 원값: {final['lr']!r}")


if __name__ == "__main__":
    main()
