import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 데이터 불러오기
# ==========================================

df = pd.read_csv("online_shoppers_intention.csv")

print("===== 원본 데이터 =====")
print("데이터 크기:", df.shape)

# ==========================================
# 2. 중복 데이터 제거
# ==========================================

duplicate_count = df.duplicated().sum()
print("\n중복 데이터:", duplicate_count)

df = df.drop_duplicates().reset_index(drop=True)

print("중복 제거 후 데이터 크기:", df.shape)

# ==========================================
# 3. 결측치 확인
# ==========================================

print("\n===== 결측치 =====")
print(df.isnull().sum().sum())

# ==========================================
# 4. X와 y 분리
# ==========================================

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

print("\n===== X / y =====")
print("X 크기:", X.shape)
print("y 크기:", y.shape)

# ==========================================
# 5. Train / Test 분할
#    80 : 20
#    Revenue 비율 유지 (stratify)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== Train / Test =====")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ==========================================
# 6. Train / Test의 Revenue 분포 확인
# ==========================================

print("\n===== Train Revenue 비율 =====")
print(y_train.value_counts(normalize=True))

print("\n===== Test Revenue 비율 =====")
print(y_test.value_counts(normalize=True))

# ==========================================
# 7. Train / Test 파일 저장
# ==========================================

train_data = X_train.copy()
train_data["Revenue"] = y_train

test_data = X_test.copy()
test_data["Revenue"] = y_test

train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

print("\n===== 파일 저장 완료 =====")
print("train.csv 저장 완료")
print("test.csv 저장 완료")