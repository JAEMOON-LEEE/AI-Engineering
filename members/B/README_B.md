# B 기준선·평가 파트

## 평가 기준

- 공식 데이터: `data/final/train_clean.csv`, `validation_final.csv`
- 입력 전처리: C1/C2 `official_v1`과 동일
  - `Weekend`를 0/1로 변환
  - `Month`, `VisitorType` 원-핫 인코딩
  - train 기준 컬럼 정렬 및 전체 입력 표준화
  - 최종 입력 26차원
- 기준선: 로지스틱 회귀, 랜덤포레스트
- 불균형 대응: `class_weight="balanced"`
- 주 지표: Recall, F1, PR-AUC
- 새 최종 MLP 분류 임계값: validation F1 기준 0.75 (11/7 확정)
- 기본 validation 실행은 test 파일을 읽지 않음. 새 동결 MLP의 test 평가는 임계값 0.75로 완료

## 파일

- `run_baseline.py`: 공식 validation 기준선 평가, 임계값 후보, 혼동행렬, ROC·PR 곡선
- `run_ablation.py`: 공식 validation에서 클래스 가중치와 `PageValues` 제외 실험
- `run_final_validation.py`: 동결 MLP와 기준선의 validation 통합 비교·임계값·오류 분석
- `run_frozen_mlp_test.py`: 합의한 임계값을 명시해 동결 MLP를 최종 test에서 1회 평가
- `B_baseline_evaluation.ipynb`: 데이터 검증, 실험 실행, 결과표·그래프 확인

## 실행

프로젝트 루트에서:

```bash
python members/B/run_baseline.py
python members/B/run_ablation.py
python members/B/run_final_validation.py
```

로컬 작업 폴더처럼 스크립트가 `scripts/`에 있으면:

```bash
python scripts/run_baseline.py
python scripts/run_ablation.py
```

새 동결 MLP는 임계값 0.75로 test 평가를 완료했다. 아래 명령은 실행 기록이며,
결과 파일이 이미 있으므로 스크립트가 재실행·덮어쓰기를 거부한다.

```bash
python members/B/run_frozen_mlp_test.py --threshold 0.75
```

`run_frozen_mlp_test.py`는 `results/B/metrics/final_test_mlp_metrics.csv`와
`results/B/predictions/final_test_mlp_predictions.csv`가 이미 있으면 덮어쓰지 않는다.

## 저장 결과

- `results/B/metrics/validation_baseline_metrics.csv`
- `results/B/metrics/validation_threshold_analysis.csv`
- `results/B/metrics/validation_threshold_candidates.csv`
- `results/B/metrics/validation_ablation_metrics.csv`
- `results/B/metrics/validation_model_comparison_metrics.csv`
- `results/B/metrics/validation_model_threshold_analysis.csv`
- `results/B/metrics/validation_model_threshold_candidates.csv`
- `results/B/predictions/validation_baseline_predictions.csv`
- `results/B/predictions/validation_all_model_predictions.csv`
- `results/B/predictions/validation_mlp_errors_at_0_5.csv`
- `results/B/metrics/final_test_mlp_metrics.csv`
- `results/B/predictions/final_test_mlp_predictions.csv`
- `results/B/figures/development_class_distribution.png`
- `results/B/figures/validation_confusion_matrices.png`
- `results/B/figures/validation_roc_pr_curves.png`
- `results/B/figures/validation_ablation_comparison.png`
- `results/B/figures/validation_all_model_confusion_matrices.png`
- `results/B/figures/validation_all_model_roc_pr_curves.png`

로컬 프로젝트에 `outputs/`가 있으면 동일한 하위 구조로 그곳에 저장한다.

## 10/4 공식 validation 결과

| 모델 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| 로지스틱 회귀 | 0.504 | 0.723 | 0.594 | 0.880 | 0.601 |
| 랜덤포레스트 | 0.647 | 0.634 | 0.640 | 0.921 | 0.728 |

- 랜덤포레스트 임계값 0.45는 Recall 0.670, F1 0.650이었다.
- 이 표는 10/4 당시 기준선 실험 기록이다. 아래 새 최종 MLP 평가에는 임계값 0.75를 적용했다.
- `PageValues` 제외 시 PR-AUC가 로지스틱 회귀 0.317, 랜덤포레스트 0.366으로 크게 하락했다.

## C1·C2 예측 결과 요청 규격

```text
row_id,split,y_true,y_probability,model_name
```

- `row_id`: `validation_final.csv`의 0부터 시작하는 행 번호
- `split`: `validation`
- `y_probability`: 구매 클래스의 예측 확률
- C1/C2가 이 형식으로 확률을 공유하면 B가 같은 기준으로 통합 혼동행렬과 PR 곡선을 작성할 수 있다.

## 11/7 새 동결 MLP 통합 validation 결과

| 모델 | 임계값 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| 로지스틱 회귀 | 0.50 | 0.504 | 0.723 | 0.594 | 0.880 | 0.601 |
| 랜덤포레스트 | 0.50 | 0.647 | 0.634 | 0.640 | 0.921 | 0.728 |
| 최종 MLP(seed 42, 56-112-56) | 0.50 | 0.459 | 0.848 | 0.596 | 0.917 | 0.702 |

- PR-AUC(AP)는 랜덤포레스트 0.728, MLP 0.702, 로지스틱 회귀 0.601 순이다.
- MLP는 기본 임계값 0.5에서 재현율이 가장 높지만 위양성이 191건으로 많다.
- 새 MLP의 validation F1 최대 후보는 0.75(Precision 0.667, Recall 0.639, F1 0.652, FP 61, FN 69)이다.
- 실비용 자료가 없어 validation F1 최대 기준인 0.75를 최종 평가 임계값으로 확정했다. 이는 실제 사업 비용의 최적점이라는 뜻은 아니다.

## 11/7 test 결과: 새 동결 MLP, 임계값 0.75

| 세션 수 | 구매 세션 수 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 2,441 | 382 | 0.676 | 0.649 | 0.662 | 0.931 | 0.729 |

- 혼동행렬: TN 1,940, FP 119, FN 134, TP 248.
- 임계값은 test 결과 확인 전에 validation에서 확정했으며, test에서 재조정하지 않았다.
- test는 새 동결 MLP만 평가했다. 로지스틱 회귀·랜덤포레스트의 test 성능과 직접 비교한 결과는 아니다.
- 같은 test를 기존 28-56-28 모델 평가에도 사용했으므로 완전히 새로 보관한 블라인드 test 결과로 표현하지 않는다. 기존 모델 test F1은 0.670으로 새 모델의 0.662보다 높았다.
- 클래스 가중치로 학습한 MLP의 출력 점수는 보정된 실제 구매 확률로 해석하지 않는다.
