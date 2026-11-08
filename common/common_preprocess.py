from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_train_val_test(train_df, val_df, test_df):
    """Train에서 전처리를 학습하고 validation/test에 동일하게 적용한다."""

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    y_train = train_df.pop("Revenue").astype(int)
    y_val = val_df.pop("Revenue").astype(int)
    y_test = test_df.pop("Revenue").astype(int)

    for df in (train_df, val_df, test_df):
        df["Weekend"] = df["Weekend"].astype(int)

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

    feature_columns = train_x.columns
    val_x = val_x.reindex(columns=feature_columns, fill_value=0)
    test_x = test_x.reindex(columns=feature_columns, fill_value=0)

    train_x = train_x.astype(float)
    val_x = val_x.astype(float)
    test_x = test_x.astype(float)

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
    project_root = Path(__file__).resolve().parents[1]
    final_dir = project_root / "data" / "final"
    train_df = pd.read_csv(final_dir / "train_clean.csv")
    val_df = pd.read_csv(final_dir / "validation_final.csv")
    test_df = pd.read_csv(final_dir / "test_final.csv")

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
