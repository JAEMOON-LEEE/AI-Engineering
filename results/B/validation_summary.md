# B 최종 평가 요약

평가 대상은 2026-11-05에 동결한 26차원 입력, 은닉층 56-112-56의 `MLP_final_seed42`다. 기준선과 MLP는 같은 validation 데이터에서 비교했다. 아래 AP는 `average_precision_score`로 계산했다.

## Validation: 공통 임계값 0.50

| 모델 | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| 로지스틱 회귀 | 0.504 | 0.723 | 0.594 | 0.880 | 0.601 |
| 랜덤포레스트 | 0.647 | 0.634 | 0.640 | 0.921 | 0.728 |
| 최종 MLP | 0.459 | 0.848 | 0.596 | 0.917 | 0.702 |

MLP는 구매자를 가장 많이 찾았지만 FP가 191건으로 많았다. 랜덤포레스트는 AP와 F1이 가장 높았다.

## 임계값 선택

MLP의 validation F1이 가장 높은 후보는 **0.75**였다. 이때 Precision 0.667, Recall 0.639, F1 0.652, FP 61건, FN 69건이다. 0.50과 비교하면 FP는 130건 줄고 FN은 40건 늘었다. 비용 자료가 없어 사업상 최적 임계값이라고 주장하지 않는다.

## Test: 동결 모델, 임계값 0.75

| 세션 | 구매 세션 | Precision | Recall | F1 | ROC-AUC | AP | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,441 | 382 | 0.676 | 0.649 | 0.662 | 0.931 | 0.729 | 1,940 | 119 | 134 | 248 |

기존 28-56-28 모델의 같은 test 결과는 임계값 0.66에서 F1 0.670, Recall 0.770, AP 0.728이었다. 새 모델의 test AP는 거의 같고 F1과 Recall은 낮으므로 새 구조가 모든 최종 분류 성능을 개선했다고 단정하지 않는다.

## 근거 파일

- `metrics/validation_model_comparison_metrics.csv`
- `metrics/validation_model_threshold_analysis.csv`
- `metrics/validation_model_threshold_candidates.csv`
- `metrics/final_test_mlp_metrics.csv`
- `predictions/validation_all_model_predictions.csv`
- `predictions/validation_mlp_errors_at_0_5.csv`
- `predictions/final_test_mlp_predictions.csv`
