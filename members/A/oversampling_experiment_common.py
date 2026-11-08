
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import recall_score, f1_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler, SMOTE

from common_preprocess import preprocess_train_val_test


# =========================
# 1. 기본 설정
# =========================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 50


# =========================
# 2. 데이터 불러오기 및 공통 전처리
# =========================

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train,
    y_val,
    y_test,
    scaler,
    feature_columns
) = preprocess_train_val_test(train_df, val_df, test_df)

y_train_np = y_train.to_numpy()
y_val_np = y_val.to_numpy()

print("===== 공통 전처리 완료 =====")
print("Train:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Test:", X_test_scaled.shape)
print("Feature 수:", len(feature_columns))
print("Train 클래스 분포 [0, 1]:", np.bincount(y_train_np))
print("Validation 클래스 분포 [0, 1]:", np.bincount(y_val_np))


# =========================
# 3. MLP 모델
# =========================

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )

    return model


# =========================
# 4. 실험 함수
# =========================

def run_experiment(
    name,
    X_train_exp,
    y_train_exp,
    class_weight=None
):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    y_train_exp = np.asarray(y_train_exp, dtype=int)

    print("학습 데이터:", X_train_exp.shape)
    print("클래스 분포 [0, 1]:", np.bincount(y_train_exp))

    tf.keras.backend.clear_session()
    model = build_model(X_train_exp.shape[1])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train_exp,
        y_train_exp,
        validation_data=(X_val_scaled, y_val_np),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=1,
    )

    # 검증 데이터로 평가
    y_pred_prob = model.predict(
        X_val_scaled,
        verbose=0
    ).ravel()

    y_pred = (y_pred_prob >= 0.5).astype(int)

    result = {
        "실험명": name,
        "Recall": recall_score(
            y_val_np, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_val_np, y_pred, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_val_np, y_pred_prob
        ),
    }

    print("\n[Validation 결과]")
    for metric in ("Recall", "F1", "PR-AUC"):
        print(f"{metric}: {result[metric]:.4f}")

    return result


# =========================
# 5. Baseline
# =========================

results = []

results.append(
    run_experiment(
        "Baseline",
        X_train_scaled,
        y_train_np
    )
)


# =========================
# 6. RandomOverSampler
# =========================

ros = RandomOverSampler(random_state=SEED)

X_ros, y_ros = ros.fit_resample(
    X_train_scaled,
    y_train_np
)

results.append(
    run_experiment(
        "RandomOverSampler",
        X_ros,
        y_ros
    )
)


# =========================
# 7. SMOTE
# =========================

smote = SMOTE(random_state=SEED)

X_smote, y_smote = smote.fit_resample(
    X_train_scaled,
    y_train_np
)

results.append(
    run_experiment(
        "SMOTE",
        X_smote,
        y_smote
    )
)


# =========================
# 8. Class Weight
# =========================

class_counts = np.bincount(y_train_np)
total_samples = len(y_train_np)
num_classes = len(class_counts)

class_weight_dict = {
    class_id: total_samples / (
        num_classes * class_counts[class_id]
    )
    for class_id in range(num_classes)
}

print("\nClass Weight:", class_weight_dict)

results.append(
    run_experiment(
        "Class Weight",
        X_train_scaled,
        y_train_np,
        class_weight=class_weight_dict
    )
)


# =========================
# 9. 최종 결과 비교 및 저장
# =========================

results_df = pd.DataFrame(results)

print("\n===== 최종 Validation 비교 =====")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    "oversampling_results_with_class_weight.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n결과 저장 완료: "
    "oversampling_results_with_class_weight.csv"
)
