import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_train_val_test(train_df, val_df, test_df):
    """
    Train/Validation/Test 공통 전처리.

    전처리 기준은 Train에 맞춰 학습하고,
    Validation/Test에는 동일한 기준을 적용한다.

    전처리:
    1. Weekend -> int
    2. Month, VisitorType -> One-Hot Encoding
    3. Train 기준으로 컬럼 정렬
    4. StandardScaler는 Train에만 fit하고 Val/Test에는 transform
    """

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Target 분리
    y_train = train_df.pop("Revenue").astype(int)
    y_val = val_df.pop("Revenue").astype(int)
    y_test = test_df.pop("Revenue").astype(int)

    # Weekend 형 변환
    for df in (train_df, val_df, test_df):
        df["Weekend"] = df["Weekend"].astype(int)

    # 범주형 변수 One-Hot Encoding
    # 모든 범주를 컬럼으로 유지
    categorical_cols = ["Month", "VisitorType"]

    train_x = pd.get_dummies(
       train_df, columns=categorical_cols, drop_first=True
    )
    val_x = pd.get_dummies(
        val_df, columns=categorical_cols, drop_first=True
    )
    test_x = pd.get_dummies(
        test_df, columns=categorical_cols, drop_first=True
    )

    # Train에 존재하는 컬럼을 기준으로 Val/Test 컬럼 맞추기
    feature_columns = train_x.columns
    val_x = val_x.reindex(columns=feature_columns, fill_value=0)
    test_x = test_x.reindex(columns=feature_columns, fill_value=0)

    # 숫자형으로 변환
    train_x = train_x.astype(float)
    val_x = val_x.astype(float)
    test_x = test_x.astype(float)

    # StandardScaler: Train에만 fit
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_x)
    x_val = scaler.transform(val_x)
    x_test = scaler.transform(test_x)

    return (
        x_train, x_val, x_test,
        y_train, y_val, y_test,
        scaler, feature_columns
    )


if __name__ == "__main__":
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("validation.csv")
    test_df = pd.read_csv("test.csv")

    (
        x_train, x_val, x_test,
        y_train, y_val, y_test,
        scaler, feature_columns
    ) = preprocess_train_val_test(train_df, val_df, test_df)

    print("===== 공통 전처리 확인 =====")
    print("Train:", x_train.shape)
    print("Validation:", x_val.shape)
    print("Test:", x_test.shape)
    print("Feature 수:", len(feature_columns))
