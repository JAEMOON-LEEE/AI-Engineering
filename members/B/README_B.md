# B 기준선·평가 파트

## 파일

- `run_baseline.py`: 로지스틱 회귀·랜덤포레스트 5겹 교차검증, 임계값 분석
- `run_ablation.py`: 클래스 가중치와 `PageValues` 특징 제외 실험
- `B_baseline_evaluation.ipynb`: 데이터 확인, 실험 실행, 결과표·그래프 확인

## 실행

프로젝트 루트에서:

```bash
python members/B/run_baseline.py
python members/B/run_ablation.py
```

기본 실행은 `data/train.csv`에서만 5겹 교차검증을 수행하며 `data/test.csv`는 사용하지 않는다.

모델, 전처리, 임계값을 모두 확정한 뒤에만 아래 명령으로 최종 테스트를 실행한다.

```bash
python members/B/run_baseline.py --evaluate-test
```

## 9/27 교차검증 기준 결과

| 모델 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| 로지스틱 회귀 | 0.523 | 0.761 | 0.620 | 0.900 | 0.646 |
| 랜덤포레스트 | 0.700 | 0.665 | 0.682 | 0.930 | 0.747 |

- 랜덤포레스트 임계값 0.45에서 Recall 0.714, F1 0.690이었다.
- 위 임계값은 교차검증 후보이며 최종 값으로 확정하지 않았다.
- `PageValues` 제외 시 두 모델의 성능이 크게 하락했다. 예측 시점에 이 특징을 사용할 수 있는지 팀 합의가 필요하다.

## C1·C2 예측 결과 규격

```text
row_id,split,y_true,y_probability,model_name
```

- `row_id`: CSV의 0부터 시작하는 행 번호
- `split`: 교차검증 예측은 `train_oof`, 최종 평가는 `test`
- `y_probability`: 구매 클래스의 예측 확률

