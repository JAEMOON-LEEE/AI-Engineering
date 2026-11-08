"""
공부 3 · Keras도 속에서는 study02와 같은 역전파를 한다

같은 가중치, 같은 배치 64개, 같은 클래스 가중치를 넣고
  (1) 손실       Keras vs numpy
  (2) 기울기     Keras GradientTape(자동 미분) vs numpy 역전파(study02의 backward)
  (3) Adam 한 걸음 뒤 가중치
를 비교한다. 차이가 float32 반올림 수준(1e-6 근처)이면 같은 계산이다.
드롭아웃은 끄고 비교한다(무작위라서 같은 마스크를 맞추기 어려움 → 마스크 원리는 study02 검산에서 확인).

사용법: python study03_keras_vs_numpy.py   (tensorflow 필요, 우리 팀 환경 그대로)
"""
import os
import sys

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "flow"))
from flow_core import load_data  # noqa: E402
import study02_our_mlp_numpy as npm  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402


def build_keras(params):
    """study02와 같은 구조, 드롭아웃 없이. 가중치를 numpy 쪽 값으로 덮어쓴다."""
    model = keras.Sequential([keras.Input(shape=(params[0]["W"].shape[0],))])
    for layer in params[:-1]:
        model.add(keras.layers.Dense(layer["W"].shape[1], activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    for k_layer, p in zip(model.layers, params):
        k_layer.set_weights([p["W"].astype("float32"), p["b"].astype("float32")])
    return model


def main():
    d = load_data()
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=d["y_tr"])
    X = d["X_tr"][:64]
    y = d["y_tr"][:64].astype("float32")
    w = np.where(y == 1, cw[1], cw[0]).astype("float32")

    rng = np.random.default_rng(0)
    params = npm.init_params(X.shape[1], npm.HIDDEN, rng)
    model = build_keras(params)

    # ---------------- Keras: 자동 미분 ----------------
    bce = keras.losses.BinaryCrossentropy()      # 기본 reduction = 가중 합 / 샘플 수
    with tf.GradientTape() as tape:
        p_k = model(X, training=False)
        loss_k = bce(y.reshape(-1, 1), p_k, sample_weight=w)
    g_k = tape.gradient(loss_k, model.trainable_variables)

    # ---------------- numpy: 손으로 쓴 역전파 ----------------
    Xf, yf = X.astype(float), y.astype(float)
    p_n, cache = npm.forward(params, Xf)          # train=False → 드롭아웃 없음
    loss_n = npm.loss_fn(p_n, yf, w)
    g_n = npm.backward(params, cache, p_n, yf, w)

    print("=== (1) 손실 ===")
    print(f"Keras {float(loss_k):.7f}   numpy {loss_n:.7f}   차이 {abs(float(loss_k) - loss_n):.1e}")

    print("\n=== (2) 기울기: Keras 자동 미분 vs numpy 역전파 ===")
    names = []
    for i in range(len(params)):
        names += [f"층 {i + 1} W", f"층 {i + 1} b"]
    flat_n = [g for layer in g_n for g in (layer["W"], layer["b"])]
    for name, gk, gn in zip(names, g_k, flat_n):
        gk = gk.numpy().reshape(gn.shape)
        print(f"{name:>7}: 최대 차이 {np.max(np.abs(gk - gn)):.1e}   (기울기 크기 {np.max(np.abs(gn)):.1e})")

    print("\n=== (3) Adam 한 걸음 (학습률 0.01/3) 뒤 가중치 ===")
    opt_k = keras.optimizers.Adam(learning_rate=npm.LR)
    opt_k.apply_gradients(zip(g_k, model.trainable_variables))
    opt_n = npm.Adam(params, npm.LR)
    opt_n.step(params, g_n)
    for i, (k_layer, p) in enumerate(zip(model.layers, params)):
        Wk = k_layer.get_weights()[0]
        print(f"층 {i + 1} W: 최대 차이 {np.max(np.abs(Wk - p['W'])):.1e}")
    print("\n→ 차이가 모두 1e-6 근처 이하면 model.fit()이 하는 일 = study02에서 손으로 쓴 것")


if __name__ == "__main__":
    main()
