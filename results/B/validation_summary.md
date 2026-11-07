# B 파트 최종 모델 평가 요약

평가 대상은 C1이 2026-11-05 동결한 26차원 입력, 은닉층 56-112-56의 `MLP_final_seed42`이다. 로지스틱 회귀와 랜덤포레스트는 같은 공식 validation 1,221세션에서 비교했다. 아래 PR-AUC 수치는 `average_precision_score`로 계산한 AP이다.

## Validation: 공통 임계값 0.50

| 모델 | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| 로지스틱 회귀 | 0.504 | 0.723 | 0.594 | 0.880 | 0.601 |
| 랜덤포레스트 | 0.647 | 0.634 | 0.640 | 0.921 | 0.728 |
| 새 최종 MLP | 0.459 | 0.848 | 0.596 | 0.917 | 0.702 |

0.50에서 MLP의 FP는 191건, FN은 29건이다. 구매자를 많이 찾지만 오탐도 많다. 랜덤포레스트는 AP와 F1이 세 모델 중 가장 높다.

## 임계값 선택

새 MLP의 validation F1 최대 후보는 **0.75**이다. 이때 Precision 0.667, Recall 0.639, F1 0.652, FP 61건, FN 69건이다. 0.50에 비해 FP는 130건 줄고 FN은 40건 늘었다. 실제 프로모션 비용과 구매 누락 비용 자료가 없으므로 사업상 최적 임계값이라고 주장하지 않는다. 임계값 0.75는 새 모델의 test 결과를 보기 전에 확정했다.

## Test: 고정된 모델과 임계값 0.75

| 세션 | 구매 세션 | Precision | Recall | F1 | ROC-AUC | AP | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,441 | 382 | 0.676 | 0.649 | 0.662 | 0.931 | 0.729 | 1,940 | 119 | 134 | 248 |

기존 28-56-28 모델의 같은 test 결과는 임계값 0.66에서 F1 0.670, Recall 0.770, AP 0.728이었다. 새 모델의 test AP는 거의 같고 F1과 Recall은 낮다. 따라서 새 구조가 최종 분류 성능까지 개선됐다고 단정하지 않는다. 이 test는 기존 모델 평가에도 사용했으므로 완전히 처음 보는 블라인드 test 결과로 표현하지 않는다. 새 test 결과를 확인한 뒤 임계값은 바꾸지 않았다.

## 결과 파일

- `results/B/metrics/validation_model_comparison_metrics.csv`
- `results/B/metrics/validation_model_threshold_analysis.csv`
- `results/B/metrics/validation_model_threshold_candidates.csv`
- `results/B/metrics/final_test_mlp_metrics.csv`
- `results/B/predictions/validation_all_model_predictions.csv`
- `results/B/predictions/validation_mlp_errors_at_0_5.csv`
- `results/B/predictions/final_test_mlp_predictions.csv`
- `results/B/figures/validation_all_model_confusion_matrices.png`
- `results/B/figures/validation_all_model_roc_pr_curves.png`
