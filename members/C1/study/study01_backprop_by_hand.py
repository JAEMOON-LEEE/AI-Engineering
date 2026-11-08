"""
공부 1 · 오류역전파를 숫자로 따라가기 (작은 신경망 2-2-1)

우리 모델(26-56-112-56-1)과 같은 재료를 아주 작게 줄였다.
  입력 2개 → 은닉층 2개(ReLU) → 출력 1개(sigmoid) → 이진 교차 엔트로피 손실
샘플 1개로 순전파 → 역전파를 한 줄씩 출력하고,
마지막에 '미분 공식으로 구한 기울기'와 '숫자로 직접 흔들어 본 기울기'가 같은지 확인한다.

사용법: python study01_backprop_by_hand.py   (numpy만 필요)
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ----------------------------------------------------------------------
# 0. 준비: 입력 1개, 정답 1개, 가중치는 손으로 정한 값
# ----------------------------------------------------------------------
x = np.array([0.5, -1.0])          # 입력 (특성 2개)
y = 1.0                            # 정답 (1 = 구매)

W1 = np.array([[0.8, -0.4],        # 입력 2 → 은닉 2  (행: 입력, 열: 은닉 노드)
               [-0.2, 0.1]])       # 은닉 노드 1은 켜지고(z>0), 노드 2는 꺼지게(z<0) 골랐다
b1 = np.array([0.0, 0.1])
W2 = np.array([0.7, -0.3])         # 은닉 2 → 출력 1
b2 = 0.05


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(W1, b1, W2, b2):
    z1 = x @ W1 + b1               # 은닉층 가중합
    h = relu(z1)                   # 은닉층 출력
    z2 = h @ W2 + b2               # 출력층 가중합
    p = sigmoid(z2)                # 구매 확률
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))   # 이진 교차 엔트로피
    return z1, h, z2, p, loss


# ----------------------------------------------------------------------
# 1. 순전파: 입력 → 출력 → 손실
# ----------------------------------------------------------------------
z1, h, z2, p, loss = forward(W1, b1, W2, b2)
print("=== 1. 순전파 ===")
print(f"z1 = x·W1 + b1        = {z1}")
print(f"h  = ReLU(z1)         = {h}     (음수는 0이 됨)")
print(f"z2 = h·W2 + b2        = {z2:.4f}")
print(f"p  = sigmoid(z2)      = {p:.4f}   (구매 확률)")
print(f"loss = -log(p)        = {loss:.4f}   (정답이 1이라 -log(p)만 남음)")

# ----------------------------------------------------------------------
# 2. 역전파: 손실에서 거꾸로, 연쇄법칙(chain rule)으로 한 칸씩
# ----------------------------------------------------------------------
print("\n=== 2. 역전파 (뒤에서 앞으로) ===")

# (a) 출력층. sigmoid + 이진 교차 엔트로피를 같이 미분하면 아주 간단해진다:
#     dL/dz2 = p - y      ← '예측 - 정답' = 오차
dz2 = p - y
print(f"dL/dz2 = p - y               = {dz2:.4f}   ← 오차. 이게 뒤로 전해진다")

# (b) 출력층 가중치: z2 = h·W2 + b2 이므로 dz2/dW2 = h
dW2 = dz2 * h
db2 = dz2
print(f"dL/dW2 = dL/dz2 × h          = {dW2}")
print(f"dL/db2 = dL/dz2              = {db2:.4f}")

# (c) 오차를 은닉층으로 보냄: dz2/dh = W2
dh = dz2 * W2
print(f"dL/dh  = dL/dz2 × W2         = {dh}   ← 가중치만큼 나눠서 책임을 돌려줌")

# (d) ReLU 통과: z1 > 0 이면 기울기 1, 아니면 0 (꺼진 노드는 책임 없음)
relu_grad = (z1 > 0).astype(float)
dz1 = dh * relu_grad
print(f"ReLU 기울기 (z1>0)           = {relu_grad}")
print(f"dL/dz1 = dL/dh × ReLU'       = {dz1}")

# (e) 첫 층 가중치: z1 = x·W1 + b1 이므로 dz1/dW1 = x
dW1 = np.outer(x, dz1)
db1 = dz1
print(f"dL/dW1 = x ⊗ dL/dz1          =\n{dW1}")
print(f"dL/db1 = dL/dz1              = {db1}")

# ----------------------------------------------------------------------
# 3. 검산: 가중치를 아주 조금(±eps) 흔들어 손실이 얼마나 변하는지 직접 잰다
#    (수치 미분). 역전파 결과와 같으면 공식이 맞은 것.
# ----------------------------------------------------------------------
print("\n=== 3. 검산 (수치 미분과 비교) ===")
eps = 1e-6
params = {"W1": W1, "b1": b1, "W2": W2}
analytic = {"W1": dW1, "b1": db1, "W2": dW2}
for name, P in params.items():
    num = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps
        lp = forward(W1, b1, W2, b2)[-1]
        P[idx] = old - eps
        lm = forward(W1, b1, W2, b2)[-1]
        P[idx] = old
        num[idx] = (lp - lm) / (2 * eps)
    diff = np.max(np.abs(num - analytic[name]))
    print(f"{name}: 역전파 {analytic[name].ravel()}  수치 {num.ravel()}  최대 차이 {diff:.1e}")

# ----------------------------------------------------------------------
# 4. 경사 하강 한 걸음: 기울기 반대 방향으로 조금 이동 → 손실이 줄어드는지
# ----------------------------------------------------------------------
lr = 0.1
W1n, b1n, W2n, b2n = W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2
_, _, _, p_new, loss_new = forward(W1n, b1n, W2n, b2n)
print("\n=== 4. 경사 하강 한 걸음 (학습률 0.1) ===")
print(f"구매 확률 {p:.4f} → {p_new:.4f}")
print(f"손실      {loss:.4f} → {loss_new:.4f}   (줄었으면 학습이 된 것)")
