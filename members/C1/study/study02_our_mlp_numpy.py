"""
공부 2 · 우리 최종 모델을 numpy로만 직접 만들기 (Keras 없이)

Keras가 model.fit() 안에서 해주는 일을 전부 손으로 쓴다.
  구조     26 → 56 → 112 → 56 → 1  (ReLU, 출력 sigmoid)
  초기화   he_normal
  손실     이진 교차 엔트로피 × 클래스 가중치 (구매 3.20 / 미구매 0.59)
  정규화   드롭아웃 0.2 (모든 은닉층)
  최적화   Adam, 학습률 0.01/3, 배치 64, 최대 30에폭
  평가     매 에폭 검증 PR-AUC, 가장 높은 에폭 기록

순서
  1) 기울기 검산: 역전파로 구한 기울기 == 수치 미분 기울기?
  2) 실제 학습: 30에폭 동안 에폭별 손실 · 검증 PR-AUC 출력
  → Keras 최종 모델(검증 PR-AUC 약 0.70)과 비슷한 수준이 나오면 같은 일을 한 것

사용법: python study02_our_mlp_numpy.py          (numpy, pandas, scikit-learn만 필요)
        python study02_our_mlp_numpy.py --check  (검산만)
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

# 데이터는 flow 단계와 똑같이 불러온다 (26차원, train에만 fit한 StandardScaler, test는 안 읽음)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flow"))
from flow_core import load_data  # noqa: E402

HIDDEN = (56, 112, 56)
DROPOUT = 0.2
LR = 0.01 / 3
BATCH = 64
EPOCHS = 30


# ======================================================================
# 모델: 가중치 만들기 (he_normal)
# ======================================================================
def init_params(n_in, hidden, rng):
    """he_normal: 표준편차 sqrt(2/입력 수)인 정규분포 (Keras는 ±2σ에서 자른 분포).
    ReLU는 절반이 0이 되니까, 신호 크기가 층마다 줄지 않도록 2를 곱해준다."""
    sizes = [n_in, *hidden, 1]
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        std = np.sqrt(2.0 / a)
        W = rng.normal(0, std, size=(a, b))
        W = np.clip(W, -2 * std, 2 * std)          # 잘린 정규분포 흉내
        params.append({"W": W, "b": np.zeros(b)})
    return params


def n_params(params):
    return sum(p["W"].size + p["b"].size for p in params)


# ======================================================================
# 순전파
# ======================================================================
def forward(params, X, rng=None, train=False, masks=None):
    """은닉층: z = a·W + b → ReLU → (학습 중이면) 드롭아웃
    출력층:  z = a·W + b → sigmoid
    역전파에 필요한 중간값(cache)을 함께 돌려준다."""
    a = X
    cache = []
    for i, layer in enumerate(params[:-1]):
        z = a @ layer["W"] + layer["b"]
        h = np.maximum(0, z)                        # ReLU
        if train and DROPOUT > 0:
            if masks is not None:
                m = masks[i]                        # 검산용: 마스크 고정
            else:
                # 역드롭아웃: 20%를 끄고 남은 값을 1/0.8배 → 평가 때는 아무것도 안 해도 됨
                m = (rng.random(h.shape) >= DROPOUT) / (1 - DROPOUT)
            out = h * m
        else:
            m = None
            out = h
        cache.append({"a_in": a, "z": z, "mask": m})
        a = out
    last = params[-1]
    z_out = (a @ last["W"] + last["b"]).ravel()
    p = 1 / (1 + np.exp(-z_out))                    # sigmoid
    cache.append({"a_in": a})
    return p, cache


def loss_fn(p, y, w):
    """클래스 가중치를 곱한 이진 교차 엔트로피. Keras와 같이 '가중 합 / 샘플 수'."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return np.sum(w * l) / len(y)


# ======================================================================
# 역전파
# ======================================================================
def backward(params, cache, p, y, w):
    """study01에서 손으로 한 것을 층 수만큼 반복할 뿐이다.
    1) 출력층 오차   dz = w·(p − y) / N
    2) 가중치 기울기 dW = (이 층 입력)ᵀ · dz,  db = dz 합
    3) 앞 층으로     da = dz · Wᵀ
    4) 드롭아웃      꺼졌던 노드는 기울기도 0 (같은 마스크를 곱함)
    5) ReLU          z ≤ 0 이었던 곳은 기울기 0
    → 2)로 돌아가 반복"""
    N = len(y)
    grads = [None] * len(params)
    dz = (w * (p - y) / N).reshape(-1, 1)           # 1) sigmoid+BCE 미분 = 예측 − 정답
    for i in range(len(params) - 1, -1, -1):
        a_in = cache[i]["a_in"] if i < len(params) - 1 else cache[-1]["a_in"]
        grads[i] = {"W": a_in.T @ dz, "b": dz.sum(axis=0)}   # 2)
        if i == 0:
            break
        da = dz @ params[i]["W"].T                  # 3)
        prev = cache[i - 1]
        if prev["mask"] is not None:
            da = da * prev["mask"]                  # 4)
        dz = da * (prev["z"] > 0)                   # 5)
    return grads


# ======================================================================
# Adam
# ======================================================================
class Adam:
    """기울기의 이동평균(m: 방향)과 기울기 제곱의 이동평균(v: 크기)을 같이 기억해
    가중치마다 보폭을 자동으로 맞춘다. Keras 기본값과 같은 β1=0.9, β2=0.999, ε=1e-7."""

    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-7):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [{k: np.zeros_like(v) for k, v in p.items()} for p in params]
        self.v = [{k: np.zeros_like(v) for k, v in p.items()} for p in params]

    def step(self, params, grads):
        self.t += 1
        # 초반에 m, v가 0 쪽으로 치우친 것을 보정하는 값을 학습률에 미리 곱해 둔다.
        # 교과서 식  lr · m̂ / (√v̂ + ε)  과 거의 같고, ε을 더하는 위치만 다르다(Keras 방식).
        alpha = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        for p, g, m, v in zip(params, grads, self.m, self.v):
            for k in p:
                m[k] = self.b1 * m[k] + (1 - self.b1) * g[k]          # 방향(기울기 평균)
                v[k] = self.b2 * v[k] + (1 - self.b2) * g[k] ** 2     # 크기(기울기 제곱 평균)
                p[k] -= alpha * m[k] / (np.sqrt(v[k]) + self.eps)


# ======================================================================
# 1) 기울기 검산
# ======================================================================
def grad_check(X, y, w, seed=0):
    rng = np.random.default_rng(seed)
    params = init_params(X.shape[1], HIDDEN, rng)
    Xb, yb, wb = X[:32].astype(float), y[:32].astype(float), w[:32]
    _, cache = forward(params, Xb, rng=rng, train=True)
    masks = [c["mask"] for c in cache[:-1]]          # 드롭아웃 마스크 고정해야 비교 가능
    p, cache = forward(params, Xb, train=True, masks=masks)
    grads = backward(params, cache, p, yb, wb)

    def L():
        return loss_fn(forward(params, Xb, train=True, masks=masks)[0], yb, wb)

    print("=== 1. 기울기 검산 (층마다 가중치 5개를 골라 수치 미분과 비교) ===")
    eps = 1e-5
    worst = 0.0
    for li, layer in enumerate(params):
        W = layer["W"]
        idxs = [tuple(rng.integers(0, s) for s in W.shape) for _ in range(5)]
        rel = []
        for idx in idxs:
            old = W[idx]
            W[idx] = old + eps
            lp = L()
            W[idx] = old - eps
            lm = L()
            W[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[li]["W"][idx]
            rel.append(abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        worst = max(worst, max(rel))
        shape = f"{W.shape[0]}→{W.shape[1]}"
        print(f"  층 {li + 1} ({shape:>7}): 상대 오차 최대 {max(rel):.1e}")
    print(f"  → {'통과' if worst < 1e-5 else '확인 필요'} (1e-5보다 작으면 역전파 공식이 맞음)\n")


# ======================================================================
# 2) 학습
# ======================================================================
def train(d, seed=42):
    rng = np.random.default_rng(seed)
    X_tr, y_tr = d["X_tr"].astype(float), d["y_tr"].astype(float)
    X_va, y_va = d["X_va"].astype(float), d["y_va"]
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=d["y_tr"])
    w_tr = np.where(y_tr == 1, cw[1], cw[0])

    params = init_params(X_tr.shape[1], HIDDEN, rng)
    opt = Adam(params, LR)
    print(f"=== 2. 학습 (numpy, 시드 {seed}) ===")
    print(f"가중치 수 {n_params(params):,}개 (Keras 최종 모델과 같아야 함: 14,281)")
    print(f"클래스 가중치  미구매 {cw[0]:.4f} / 구매 {cw[1]:.4f}")
    print(f"{'에폭':>4} {'학습 손실':>9} {'검증 PR-AUC':>11}")
    best = (-1, 0)
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_tr))          # 매 에폭 섞기
        tot = 0.0
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            p, cache = forward(params, X_tr[b], rng=rng, train=True)   # 순전파
            tot += loss_fn(p, y_tr[b], w_tr[b]) * len(b)
            grads = backward(params, cache, p, y_tr[b], w_tr[b])       # 역전파
            opt.step(params, grads)                                    # 가중치 갱신
        p_va, _ = forward(params, X_va)              # 평가: 드롭아웃 끔
        pr = average_precision_score(y_va, p_va)
        mark = ""
        if pr > best[0]:
            best = (pr, ep)
            mark = "  ← 최고"
        print(f"{ep:>4} {tot / len(y_tr):>9.4f} {pr:>11.4f}{mark}")
    print(f"\n최고 검증 PR-AUC {best[0]:.4f} (에폭 {best[1]}), {time.time() - t0:.0f}초")
    print("Keras 최종 모델: 0.7053 ± 0.0039 (시드 42/43/44)")
    print("※ 난수 생성 방식이 Keras와 달라 숫자가 똑같지는 않다. 비슷한 수준이면 같은 일을 한 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="검산만")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    d = load_data()
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=d["y_tr"])
    w = np.where(d["y_tr"] == 1, cw[1], cw[0])
    grad_check(d["X_tr"], d["y_tr"], w)
    if not a.check:
        train(d, a.seed)


if __name__ == "__main__":
    main()
