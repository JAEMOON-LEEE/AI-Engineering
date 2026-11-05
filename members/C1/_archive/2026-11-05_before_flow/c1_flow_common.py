"""
새 흐름용 공통 학습·기록.

데이터·옵티마이저·결정론은 c1_ver2_adam.py를 읽고만 쓴다.
최고 에폭은 Keras val_pr_auc가 가장 높은 에폭의 가중치로 되돌린 뒤
sklearn average_precision_score로 보고한다 (c1_ver2_adam.run_one과 같음).

에폭마다 predict를 호출하지 않는다. 가중치만 복사했다가 학습이 끝난 뒤
sklearn 곡선을 계산한다 (난수 상태가 다음 에폭 배치 순서를 바꾸지 않게).
"""

from __future__ import annotations

import json
import os
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

C1_DIR = os.path.dirname(os.path.abspath(__file__))
if C1_DIR not in sys.path:
    sys.path.insert(0, C1_DIR)

import c1_ver2_adam as v2
from c1_ver2_adam import keras, layers, tf

RUNS_CSV = os.path.join(C1_DIR, "c1_flow_runs.csv")
SEEDS = list(v2.SEEDS)
EPOCH_COLS = [
    "cfg", "seed", "epoch",
    "val_pr_keras", "val_pr_sk", "train_pr", "loss", "val_loss",
    "recall", "precision", "f1", "pos_rate",
    "best_epoch", "n_epochs",
    "report_val_pr", "report_train_pr", "report_recall",
    "report_precision", "report_f1", "report_pos_rate",
    "n_params", "seconds",
]


class EpochSnapshot(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.weights = []
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append([w.copy() for w in self.model.get_weights()])
        self.logs.append({k: float(v) for k, v in (logs or {}).items()})


def default_cfg(**updates):
    cfg = dict(
        hidden=(28, 56, 28),
        dropout=0.0,
        drop_where="none",
        optimizer="Adam",
        lr=0.01,
        batch=128,
        max_epochs=30,
        early_stop=0,
        class_weight="balanced",
    )
    cfg.update(updates)
    if not isinstance(cfg["hidden"], tuple):
        cfg["hidden"] = tuple(cfg["hidden"])
    cfg["dropout"] = float(cfg["dropout"])
    cfg["lr"] = float(cfg["lr"])
    cfg["batch"] = int(cfg["batch"])
    cfg["max_epochs"] = int(cfg["max_epochs"])
    cfg["early_stop"] = int(cfg["early_stop"])
    return cfg


def key_of(cfg):
    payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(cfg.items())}
    return json.dumps(payload)


def load_data():
    return v2.load_data()


def class_weight_of(y, mode):
    if mode in (None, "none"):
        return None
    if mode != "balanced":
        raise ValueError(f"지원하지 않는 class_weight: {mode}")
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


def build(cfg, input_dim):
    hidden = tuple(cfg["hidden"])
    seq = [layers.Input(shape=(input_dim,))]
    for i, units in enumerate(hidden):
        seq.append(layers.Dense(units, activation="relu", kernel_initializer="he_normal"))
        is_last = i == len(hidden) - 1
        where = cfg.get("drop_where", "none")
        if cfg["dropout"] > 0 and (
            where == "all" or (where == "last" and is_last)
        ):
            seq.append(layers.Dropout(cfg["dropout"]))
    seq.append(layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(seq)
    model.compile(
        optimizer=v2.make_optimizer(cfg["optimizer"], cfg["lr"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def _threshold_metrics(y_true, p):
    pred = (p >= 0.5).astype(int)
    return dict(
        recall=float(recall_score(y_true, pred, zero_division=0)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        pos_rate=float(pred.mean()),
    )


def _ap(y_true, p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        return float("nan")
    return float(average_precision_score(y_true, p))


def _read_cache():
    if not os.path.isfile(RUNS_CSV):
        return pd.DataFrame(columns=EPOCH_COLS)
    return pd.read_csv(RUNS_CSV)


def cached_rows(cfg, seed, cache=None):
    df = _read_cache() if cache is None else cache
    if df.empty:
        return None
    hit = df[(df["cfg"] == key_of(cfg)) & (df["seed"] == seed)]
    return None if hit.empty else hit.sort_values("epoch")


def run_one(cfg, seed, d):
    """한 시드 학습. 같은 cfg+seed가 csv에 있으면 학습하지 않는다."""
    cfg = default_cfg(**cfg)
    hit = cached_rows(cfg, seed)
    if hit is not None:
        print(f"  [캐시] seed {seed}", flush=True)
        return hit

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    cw = class_weight_of(d["y_tr"], cfg["class_weight"])
    snap = EpochSnapshot()
    callbacks = [snap]
    if cfg["early_stop"]:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=cfg["early_stop"], restore_best_weights=True,
        ))
    else:
        callbacks.append(v2.BestWeights())

    model = build(cfg, d["X_tr"].shape[1])
    t0 = time.perf_counter()
    history = model.fit(
        d["X_tr"], d["y_tr"],
        validation_data=(d["X_va"], d["y_va"]),
        epochs=cfg["max_epochs"],
        batch_size=cfg["batch"],
        class_weight=cw,
        callbacks=callbacks,
        verbose=0,
    ).history
    seconds = round(time.perf_counter() - t0, 1)

    keras_curve = [float(v) for v in history["val_pr_auc"]]
    best_epoch = int(np.argmax(keras_curve)) + 1
    n_epochs = len(keras_curve)
    n_params = int(model.count_params())

    p_va = model.predict(d["X_va"], verbose=0).ravel()
    p_tr = model.predict(d["X_tr"], verbose=0).ravel()
    report = dict(
        report_val_pr=_ap(d["y_va"], p_va),
        report_train_pr=_ap(d["y_tr"], p_tr),
        **{f"report_{k}": v for k, v in _threshold_metrics(d["y_va"], p_va).items()},
    )

    rows = []
    k = key_of(cfg)
    for epoch, (weights, logs) in enumerate(zip(snap.weights, snap.logs), start=1):
        model.set_weights(weights)
        p_tr_e = model.predict(d["X_tr"], verbose=0).ravel()
        p_va_e = model.predict(d["X_va"], verbose=0).ravel()
        m = _threshold_metrics(d["y_va"], p_va_e)
        rows.append(dict(
            cfg=k, seed=seed, epoch=epoch,
            val_pr_keras=keras_curve[epoch - 1],
            val_pr_sk=_ap(d["y_va"], p_va_e),
            train_pr=_ap(d["y_tr"], p_tr_e),
            loss=float(logs["loss"]),
            val_loss=float(logs["val_loss"]),
            recall=m["recall"], precision=m["precision"], f1=m["f1"], pos_rate=m["pos_rate"],
            best_epoch=best_epoch, n_epochs=n_epochs,
            n_params=n_params, seconds=seconds,
            **report,
        ))
    out = pd.DataFrame(rows)[EPOCH_COLS]
    os.makedirs(C1_DIR, exist_ok=True)
    out.to_csv(RUNS_CSV, mode="a", header=not os.path.isfile(RUNS_CSV),
               index=False, encoding="utf-8-sig")
    return out


def run_seeds(cfg, d, seeds=None):
    cfg = default_cfg(**cfg)
    frames = [run_one(cfg, seed, d) for seed in (seeds or SEEDS)]
    return pd.concat(frames, ignore_index=True)


def seed_reports(df):
    """시드별 Keras 최고 에폭으로 복원한 보고 점수."""
    rows = []
    for seed, sub in df.groupby("seed"):
        r = sub.iloc[0]
        rows.append(dict(
            seed=int(seed),
            val_pr=float(r["report_val_pr"]),
            train_pr=float(r["report_train_pr"]),
            recall=float(r["report_recall"]),
            precision=float(r["report_precision"]),
            f1=float(r["report_f1"]),
            pos_rate=float(r["report_pos_rate"]),
            best_epoch=int(r["best_epoch"]),
            n_epochs=int(r["n_epochs"]),
        ))
    return pd.DataFrame(rows).sort_values("seed")


def combo_cells(label, reports, max_epochs=None):
    val = reports["val_pr"].to_numpy(dtype=float)
    train = reports["train_pr"].to_numpy(dtype=float)
    val_mean, train_mean = float(val.mean()), float(train.mean())
    cap = max_epochs if max_epochs is not None else int(round(float(reports["n_epochs"].mean())))
    return [
        label,
        f"{val_mean:.4f}±{float(val.std(ddof=1)):.4f}",
        f"{train_mean:.3f}",
        f"{train_mean - val_mean:+.3f}",
        f"{float(reports['recall'].mean()):.3f}",
        f"{float(reports['precision'].mean()):.4f}",
        f"{float(reports['pos_rate'].mean()):.3f}",
        f"{int(round(float(reports['best_epoch'].mean())))}/{int(round(float(reports['n_epochs'].mean())))}",
    ], val_mean, float(val.std(ddof=1))


def disp_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def pad(text, width, align="left"):
    gap = width - disp_width(text)
    if gap <= 0:
        return text
    if align == "right":
        return " " * gap + text
    return text + " " * gap


HEADERS = ["후보", "PR-AUC", "학습PR", "차이", "재현율", "정밀도", "양성예측", "최고/전체에폭"]
ALIGNS = ["left"] + ["right"] * 7


def render_table(rows):
    widths = [
        max(disp_width(HEADERS[i]), *(disp_width(row[i]) for row in rows))
        for i in range(len(HEADERS))
    ]
    lines = []
    for cells in (HEADERS, *rows):
        lines.append("  ".join(pad(cells[i], widths[i], ALIGNS[i]) for i in range(len(cells))))
    return "\n".join(lines), widths
