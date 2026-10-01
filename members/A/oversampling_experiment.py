import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler, SMOTE


# =========================
# 1. 기본 설정
# =========================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

HIDDEN_UNITS = (64, 32)
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 50


# =========================
# 2. 데이터 불러오기
# =========================

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

print("Train:", train_df.shape)
print("Validation:", val_df.shape)


# =========================
# 3. 전처리
# =========================

def preprocess(df, ref_columns=None):
    df = df.copy()

    # Weekend → 0/1
    df["Weekend"] = df["Weekend"].astype(int)

    # Target
    y = df["Revenue"].astype(int)

    # 입력 데이터
    X = df.drop(columns=["Revenue"])

    # 범주형 변수 원-핫 인코딩
    X = pd.get_dummies(
        X,
        columns=["Month", "VisitorType"],
        drop_first=True
    )

    # Train의 컬럼 구조에 맞춤
    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0)

    return X, y


X_train, y_train = preprocess(train_df)
X_val, y_val = preprocess(
    val_df,
    ref_columns=X_train.columns
)

print("전처리 후 Train:", X_train.shape)
print("전처리 후 Validation:", X_val.shape)


# =========================
# 4. 스케일링
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# =========================
# 5. MLP 모델
# =========================

def build_model(input_dim):

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(
                curve="PR",
                name="pr_auc"
            )
        ]
    )

    return model


# =========================
# 6. 실험 함수
# =========================

def run_experiment(name, X_train_exp, y_train_exp):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("학습 데이터:", X_train_exp.shape)
    print(
        "클래스 분포:",
        np.bincount(y_train_exp)
    )

    model = build_model(X_train_exp.shape[1])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train_exp,
        y_train_exp,
        validation_data=(X_val_scaled, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    # Validation 예측
    y_pred_prob = model.predict(
        X_val_scaled,
        verbose=0
    ).ravel()

    y_pred = (y_pred_prob >= 0.5).astype(int)

    # 평가
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    pr_auc = average_precision_score(
        y_val,
        y_pred_prob
    )

    print("\n[Validation 결과]")
    print(f"Recall : {recall:.4f}")
    print(f"F1     : {f1:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    return recall, f1, pr_auc


# =========================
# 7. Baseline
# =========================

baseline_result = run_experiment(
    "① Baseline",
    X_train_scaled,
    y_train.to_numpy()
)


# =========================
# 8. RandomOverSampler
# =========================

ros = RandomOverSampler(
    random_state=SEED
)

X_ros, y_ros = ros.fit_resample(
    X_train_scaled,
    y_train.to_numpy()
)

ros_result = run_experiment(
    "② RandomOverSampler",
    X_ros,
    y_ros
)


# =========================
# 9. SMOTE
# =========================

smote = SMOTE(
    random_state=SEED
)

X_smote, y_smote = smote.fit_resample(
    X_train_scaled,
    y_train.to_numpy()
)

smote_result = run_experiment(
    "③ SMOTE",
    X_smote,
    y_smote
)


# =========================
# 10. 최종 비교
# =========================

print("\n")
print("=" * 60)
print("최종 실험 결과")
print("=" * 60)

print(
    f"Baseline          "
    f"Recall={baseline_result[0]:.4f} "
    f"F1={baseline_result[1]:.4f} "
    f"PR-AUC={baseline_result[2]:.4f}"
)

print(
    f"RandomOverSampler "
    f"Recall={ros_result[0]:.4f} "
    f"F1={ros_result[1]:.4f} "
    f"PR-AUC={ros_result[2]:.4f}"
)

print(
    f"SMOTE             "
    f"Recall={smote_result[0]:.4f} "
    f"F1={smote_result[1]:.4f} "
    f"PR-AUC={smote_result[2]:.4f}"
)