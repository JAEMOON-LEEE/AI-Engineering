"""
C1 담당 - MLP 기본 골격 + 학습/평가 루프 + 실험 로그 자동 기록
- 9/22: MLP(입력->은닉2층->출력) + 학습/평가 루프
- 9/23: 첫 학습 + 재현율/F1 측정, 로그 기록 시작
- 9/27: 엑셀 로그 자동 기록 추가

전제:
- A가 나눠준 train.csv / test.csv 사용 (train_test_split 80:20, stratify=Revenue, seed=42)
- B의 평가 스크립트 아직 없어서, train.csv를 다시 train/val로 "임시" 분할해서 사용
  -> B 스크립트 오면 3번 섹션만 교체할 것
- 전처리도 "임시": Weekend를 int로, Month/VisitorType은 원-핫
  -> A의 공통 전처리 오면 이 부분 교체할 것
- 같은 폴더에 있는 experiment_log_C1.xlsx(팀 공통 양식)에 결과를 자동으로 한 줄씩 추가함
"""

import os
from datetime import date

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, average_precision_score, classification_report
import openpyxl

# ---- 경로 설정 (이 스크립트와 같은 폴더 기준) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
LOG_PATH = os.path.join(BASE_DIR, "experiment_log_C1.xlsx")  # 팀 공통 엑셀 양식

# ---- 0. 시드 고정 ----
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- 1. 데이터 불러오기 (A가 나눠준 train/test) ----
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print("train:", train_df.shape, "test:", test_df.shape)


# ---- 2. 임시 전처리 (팀 공통 전처리 오면 이 함수 통째로 교체) ----
PREP_VERSION = "temp_v1(원-핫, A공통전처리 전)"


def preprocess(df, ref_columns=None):
    df = df.copy()
    df["Weekend"] = df["Weekend"].astype(int)
    y = df["Revenue"].astype(int) if "Revenue" in df.columns else None
    X = df.drop(columns=["Revenue"]) if "Revenue" in df.columns else df
    X = pd.get_dummies(X, columns=["Month", "VisitorType"], drop_first=True)
    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0)
    return X, y


X_train_full, y_train_full = preprocess(train_df)
X_test, y_test = preprocess(test_df, ref_columns=X_train_full.columns)

# ---- 3. 임시 train/val 분할 (B 스크립트 오면 이 부분 교체) ----
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, random_state=SEED, stratify=y_train_full
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
print("train/val/test shape:", X_train_s.shape, X_val_s.shape, X_test_s.shape)


# ---- 4. MLP 기본 골격 (입력 -> 은닉 2층 -> 출력) ----
HIDDEN_UNITS = (64, 32)


def build_model(input_dim, hidden_units=HIDDEN_UNITS):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units[0], activation="relu"),
        layers.Dense(hidden_units[1], activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


model = build_model(X_train_s.shape[1])
model.summary()

# ---- 5. 학습 ----
EPOCHS = 50
BATCH_SIZE = 32

history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

# ---- 6. 평가: 정확도(X) 대신 재현율/F1/PR-AUC로 ----
y_val_pred_prob = model.predict(X_val_s).ravel()
y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
pr_auc = average_precision_score(y_val, y_val_pred_prob)

print(f"\n[검증셋] Recall: {recall:.4f}  F1: {f1:.4f}  PR-AUC: {pr_auc:.4f}")
print(classification_report(y_val, y_val_pred, target_names=["No Purchase", "Purchase"]))


# ---- 7. 실험 로그 엑셀에 자동 기록 ----
def log_to_excel(log_path, entry, sheet_name="실험로그"):
    """팀 공통 엑셀 양식(실험로그 시트) 맨 아래 빈 줄에 한 행 추가."""
    wb = openpyxl.load_workbook(log_path)
    ws = wb[sheet_name]

    # A열이 비어있는 첫 행 찾기 (헤더는 1행)
    next_row = 2
    while ws.cell(row=next_row, column=1).value is not None:
        next_row += 1

    exp_id = f"C1-{next_row - 1:03d}"

    row_values = [
        exp_id,
        entry["date"],
        entry["stage"],
        entry["framework"],
        entry["prep_version"],
        entry["input_dim"],
        entry["hidden_layers"],
        entry["nodes"],
        entry["optimizer"],
        entry["lr"],
        entry["batch_size"],
        entry["max_epochs"],
        entry["early_stop"],
        entry["seed"],
        entry["recall"],
        entry["f1"],
        entry["pr_auc"],
        entry["memo"],
    ]
    for col, value in enumerate(row_values, start=1):
        ws.cell(row=next_row, column=col, value=value)

    wb.save(log_path)
    print(f"[로그 저장 완료] {exp_id} -> {log_path}")
    return exp_id


log_entry = {
    "date": date.today().isoformat(),
    "stage": "baseline",
    "framework": f"tensorflow(keras) {tf.__version__}",
    "prep_version": PREP_VERSION,
    "input_dim": X_train_s.shape[1],
    "hidden_layers": len(HIDDEN_UNITS),
    "nodes": str(HIDDEN_UNITS),
    "optimizer": "adam",
    "lr": "default(1e-3)",
    "batch_size": BATCH_SIZE,
    "max_epochs": EPOCHS,
    "early_stop": "없음(고정 50 epoch)",
    "seed": SEED,
    "recall": round(float(recall), 4),
    "f1": round(float(f1), 4),
    "pr_auc": round(float(pr_auc), 4),
    "memo": "baseline, val_loss는 epoch 6~7 이후 과적합 시작(C2 조기종료 기준 참고용)",
}

log_to_excel(LOG_PATH, log_entry)