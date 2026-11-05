"""
9단계 · 에폭 확인 (최종 설정에서 최대 에폭만 변경)

8단계까지 정한 설정(56-112-56, Adam 0.01/3, 배치 64, 드롭아웃 0.2, 클래스 가중치)에서
최고 에폭이 시드별 30/30/27로 끝에 붙어 있어 '30에폭이 부족한가'를 확인한다.
후보: 최대 에폭 30 / 50 / 100 (모두 학습 후 검증 PR-AUC 최고 에폭 가중치로 평가).
선택 규칙: 1위 (다른 단계와 같음, 실행 전에 정함).
30은 캐시 재사용, 50·100은 처음 한 번 학습(시드 3개 × 2 = 6번).

사용법: python step09_epoch.py
"""
import flow_core as fc
from step08_lr_recheck import compute as prev

EPOCHS = [30, 50, 100]
EXPECTED = {"최대 30에폭": 0.7053}
COLS = ["pr", "train", "gap", "recall", "precision", "best", "drop", "sec", "status"]


def compute():
    _, base, _ = prev()
    cands = [(f"최대 {e}에폭", fc.cfg_of(**dict(base, max_epochs=e))) for e in EPOCHS]
    res = fc.evaluate(cands)
    chosen, line = fc.judge(res, f"최대 {base['max_epochs']}에폭", rule="best")
    return res, chosen["cfg"], line


def main():
    res, chosen, line = compute()
    fc.header("9단계 · 에폭 확인", fc.cfg_line(chosen, skip=("max_epochs",)))
    print(fc.table(res, COLS))
    print(fc.seed_table(res))
    print("검증 하락 = 최고 에폭 검증PR − 마지막 에폭 검증PR")
    print(line)
    fc.check_values(res, EXPECTED)
    print("\n=== 최종 설정 (동결 전) ===")
    print(fc.cfg_line(chosen))
    print(f"학습률 원값: {chosen['lr']!r}")
    return chosen


if __name__ == "__main__":
    main()
