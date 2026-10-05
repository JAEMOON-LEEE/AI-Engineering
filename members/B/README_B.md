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
- 분류 임계값: 기본 0.5. validation에서 후보만 비교하고 임의로 확정하지 않음
- test는 기본 실행에서 파일 자체를 읽지 않음

## 파일

- `run_baseline.py`: 공식 validation 기준선 평가, 임계값 후보, 혼동행렬, ROC·PR 곡선
- `run_ablation.py`: 공식 validation에서 클래스 가중치와 `PageValues` 제외 실험
- `B_baseline_evaluation.ipynb`: 데이터 검증, 실험 실행, 결과표·그래프 확인

## 실행

프로젝트 루트에서:

```bash
python members/B/run_baseline.py
python members/B/run_ablation.py
```

로컬 작업 폴더처럼 스크립트가 `scripts/`에 있으면:

```bash
python scripts/run_baseline.py
python scripts/run_ablation.py
```

모든 모델과 임계값을 팀에서 확정한 뒤에만 최종 테스트를 실행한다.

```bash
python members/B/run_baseline.py --evaluate-test
```

## 저장 결과

- `results/B/metrics/validation_baseline_metrics.csv`
- `results/B/metrics/validation_threshold_analysis.csv`
- `results/B/metrics/validation_threshold_candidates.csv`
- `results/B/metrics/validation_ablation_metrics.csv`
- `results/B/predictions/validation_baseline_predictions.csv`
- `results/B/figures/development_class_distribution.png`
- `results/B/figures/validation_confusion_matrices.png`
- `results/B/figures/validation_roc_pr_curves.png`
- `results/B/figures/validation_ablation_comparison.png`

로컬 프로젝트에 `outputs/`가 있으면 동일한 하위 구조로 그곳에 저장한다.

## 10/4 공식 validation 결과

| 모델 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| 로지스틱 회귀 | 0.504 | 0.723 | 0.594 | 0.880 | 0.601 |
| 랜덤포레스트 | 0.647 | 0.634 | 0.640 | 0.921 | 0.728 |

- 랜덤포레스트 임계값 0.45는 Recall 0.670, F1 0.650이었다.
- 임계값은 validation 후보이며 팀의 비용 기준 합의 전에는 확정하지 않는다.
- `PageValues` 제외 시 PR-AUC가 로지스틱 회귀 0.317, 랜덤포레스트 0.366으로 크게 하락했다.

## C1·C2 예측 결과 요청 규격

```text
row_id,split,y_true,y_probability,model_name
```

- `row_id`: `validation_final.csv`의 0부터 시작하는 행 번호
- `split`: `validation`
- `y_probability`: 구매 클래스의 예측 확률
- C1/C2가 이 형식으로 확률을 공유하면 B가 같은 기준으로 통합 혼동행렬과 PR 곡선을 작성할 수 있다.
