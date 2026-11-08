# B 기준선·성능 평가

## 평가 규칙

- 공식 분할: `data/final/train_clean.csv`, `validation_final.csv`, `test_final.csv`
- 공통 전처리: `Weekend` 정수 변환, `Month`·`VisitorType` 원-핫 인코딩, train 기준 표준화, 입력 26차원
- 기준선: 로지스틱 회귀, 랜덤포레스트 (`class_weight="balanced"`)
- 모델 비교: 같은 validation과 같은 전처리 조건 사용
- 모델 비교 지표: Precision, Recall, F1, ROC-AUC, AP
- 최종 MLP 임계값: validation F1 최대 기준 **0.75**
- test 결과로 모델이나 임계값을 다시 조정하지 않음

## 실행 파일

- `run_baseline.py`: 두 기준선의 validation 평가와 그래프 생성
- `run_ablation.py`: 클래스 가중치와 PageValues 제거에 따른 기준선 변화 확인용 보조 실험
- `run_final_validation.py`: 기준선과 동결 MLP의 validation 통합 비교, 임계값 후보와 오류 분석
- `run_frozen_mlp_test.py`: 확정된 임계값으로 동결 MLP의 test 평가
- `B_baseline_evaluation.ipynb`: 실행 절차 확인용 노트북

프로젝트 루트에서 다음 순서로 실행한다.

```bash
python members/B/run_baseline.py
python members/B/run_ablation.py
python members/B/run_final_validation.py
```

최종 test 평가는 이미 완료됐다. 결과 파일이 있으면 test 스크립트가 덮어쓰기를 막는다.

```bash
python members/B/run_frozen_mlp_test.py --threshold 0.75
```

## Validation 결과

공통 임계값 0.50에서 비교한 결과다. AP는 특정 임계값 하나가 아니라 전체 PR 관계를 요약한다.

| 모델 | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| 로지스틱 회귀 | 0.504 | 0.723 | 0.594 | 0.880 | 0.601 |
| 랜덤포레스트 | 0.647 | 0.634 | 0.640 | 0.921 | 0.728 |
| 최종 MLP (seed 42) | 0.459 | 0.848 | 0.596 | 0.917 | 0.702 |

- 랜덤포레스트는 AP와 F1이 가장 높다.
- MLP는 Recall이 가장 높지만 FP가 191건으로 많다.
- MLP의 validation F1 최대 후보는 0.75다: Precision 0.667, Recall 0.639, F1 0.652, FP 61, FN 69.
- 비용 자료가 없어 0.75를 사업상 최적값으로 해석하지 않는다.

## 최종 test 결과

동결된 `MLP_final_seed42`와 임계값 0.75를 test 2,441세션에 적용했다.

| Precision | Recall | F1 | ROC-AUC | AP | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.676 | 0.649 | 0.662 | 0.931 | 0.729 | 1,940 | 119 | 134 | 248 |

- 실제 구매 세션은 382개이며 248개를 찾고 134개를 놓쳤다.
- test에서는 최종 MLP만 평가했으므로 기준선과 test 성능을 직접 비교하지 않는다.
- 같은 test가 이전 모델 평가에도 사용됐으므로 완전히 새로운 블라인드 test라고 표현하지 않는다.
- 모델 출력은 클래스 가중치의 영향을 받으므로 보정된 실제 구매 확률로 해석하지 않는다.

## 결과 위치

- `results/B/metrics/`: 모델 비교, 임계값 후보, 최종 test 지표
- `results/B/predictions/`: validation·test 예측과 MLP 오류 행
- `results/B/figures/`: 혼동행렬과 ROC·PR 곡선
- `results/B/validation_summary.md`: 발표에 사용하는 최종 평가 요약

`run_ablation.py`의 PageValues 제거 실험은 민감도 확인용 기록이며 최종 MLP 또는 임계값 0.75의 선택 근거가 아니다.
