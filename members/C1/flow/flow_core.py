"""
새 흐름(2026-11-05) 공통 모듈 — 데이터, 모델, 학습, 결과 캐시, 표 출력.

이 파일 하나로 단계별 스크립트(step00 ~ step09)가 돌아간다.
예전 c1_ver2_adam.py / c1_flow_common.py 를 import 하지 않도록 필요한 부분만 그대로 옮겼다.
학습 절차는 c1_flow_common.run_one 과 한 줄도 다르지 않게 유지했다(같은 시드면 같은 숫자).

평가 방식
  - 매 에폭 가중치를 복사해 두고(EpochSnapshot), 학습이 끝나면
    Keras val_pr_auc 가 가장 높았던 에폭의 가중치로 되돌린다(BestWeights).
  - 보고 점수는 그 가중치로 계산한 sklearn average_precision_score(검증 PR-AUC).
  - 고르는 기준(Keras)과 보고 점수(sklearn)를 분리해 낙관 편향을 줄인다.

결과 캐시
  - flow_runs.csv 한 파일에 (설정, 시드, 에폭)별로 저장한다. 같은 설정·시드가 있으면 학습하지 않는다.
  - 설정 키(key_of)는 예전 c1_flow_runs.csv 와 같아서, 예전 결과를 그대로 이어 쓴다.

TensorFlow 는 실제로 학습이 필요할 때만 불러온다(캐시만 읽을 때는 TF 없이도 실행됨).
"""

from __future__ import annotations

import json
import os
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

FLOW_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(FLOW_DIR, "..", "..", ".."))
DATA_DIR = os.path.join(REPO_DIR, "data", "final")
RUNS_CSV = os.path.join(FLOW_DIR, "flow_runs.csv")
FIG_DIR = os.path.join(FLOW_DIR, "figures")
SEEDS = [42, 43, 44]

# 학습률 0.0033 이 아니라 정확히 0.01/3 이다(2026-11-05 확인: 0.0033 으로 넣으면 결과가 다름).
LR_THIRD = 0.01 / 3
LR_TEXT = {5e-05: "0.00005", 0.001: "0.001", LR_THIRD: "0.01/3", 0.0033: "0.0033(잘못된 값)",
           0.01: "0.01", 0.03: "0.03", 0.1: "0.1"}

CACHE_ONLY = os.environ.get("FLOW_CACHE_ONLY") == "1"   # 1이면 학습하지 않고 캐시만 사용

EPOCH_COLS = [
    "cfg", "seed", "epoch",
    "val_pr_keras", "val_pr_sk", "train_pr", "loss", "val_loss",
    "recall", "precision", "f1", "pos_rate",
    "best_epoch", "n_epochs",
    "report_val_pr", "report_train_pr", "report_recall",
    "report_precision", "report_f1", "report_pos_rate",
    "n_params", "seconds",
]


# ======================================================================
# 설정
# ======================================================================
def cfg_of(**updates):
    """설정 dict. 기본값은 0단계 초기값과 같은 틀(값은 각 단계에서 덮어씀)."""
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
    cfg["hidden"] = tuple(int(u) for u in cfg["hidden"])
    cfg["dropout"] = float(cfg["dropout"])
    if cfg["dropout"] > 0 and cfg["drop_where"] == "none":
        cfg["drop_where"] = "all"
    if cfg["dropout"] == 0:
        cfg["drop_where"] = "none"
    cfg["lr"] = float(cfg["lr"])
    cfg["batch"] = int(cfg["batch"])
    cfg["max_epochs"] = int(cfg["max_epochs"])
    cfg["early_stop"] = int(cfg["early_stop"])
    return cfg


def key_of(cfg):
    payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(cfg.items())}
    return json.dumps(payload)


def lr_text(lr):
    for k, v in LR_TEXT.items():
        if lr == k:
            return v
    return f"{lr:g}"


def hidden_text(hidden):
    return "-".join(str(u) for u in hidden)


def cfg_line(cfg, skip=()):
    """터미널 조건 한 줄."""
    parts = []
    if "hidden" not in skip:
        parts.append(f"({','.join(str(u) for u in cfg['hidden'])})")
    if "optimizer" not in skip:
        opt = "SGD+NAG" if cfg["optimizer"] == "SGD" else cfg["optimizer"]
        parts.append(opt if "lr" in skip else f"{opt} {lr_text(cfg['lr'])}")
    elif "lr" not in skip:
        parts.append(f"학습률 {lr_text(cfg['lr'])}")
    if "batch" not in skip:
        parts.append(f"배치 {cfg['batch']}")
    if "max_epochs" not in skip:
        parts.append(f"최대 {cfg['max_epochs']}에폭")
    if "dropout" not in skip:
        parts.append(f"드롭아웃 {cfg['dropout']:g}" + ("(모든 은닉층)" if cfg["dropout"] > 0 else ""))
    if "early_stop" not in skip:
        parts.append(f"조기종료 p{cfg['early_stop']}" if cfg["early_stop"] else "조기종료 없음")
    if "class_weight" not in skip:
        parts.append({"balanced": "클래스 가중치", "none": "불균형 처리 없음",
                      "under_1to4": "언더샘플링 1:4"}[cfg["class_weight"]])
    return ", ".join(parts)


# ======================================================================
# 데이터 (c1_ver2_adam.load_data 와 동일, 26차원, test 는 읽지 않음)
# ======================================================================
def preprocess(df, ref_columns=None):
    df = df.copy()
    df["Weekend"] = df["Weekend"].astype(int)
    y = df["Revenue"].astype(int)
    X = df.drop(columns=["Revenue"])
    if ref_columns is None:
        X = pd.get_dummies(X, columns=["Month", "VisitorType"], drop_first=True)
    else:
        X = pd.get_dummies(X, columns=["Month", "VisitorType"], drop_first=False)
        X = X.reindex(columns=ref_columns, fill_value=0)
    return X, y


_DATA = None


def load_data():
    global _DATA
    if _DATA is None:
        tr = pd.read_csv(os.path.join(DATA_DIR, "train_clean.csv"))
        va = pd.read_csv(os.path.join(DATA_DIR, "validation_final.csv"))
        X_tr, y_tr = preprocess(tr)
        X_va, y_va = preprocess(va, ref_columns=X_tr.columns)
        sc = StandardScaler()
        _DATA = dict(
            X_tr=sc.fit_transform(X_tr.astype(float)).astype("float32"), y_tr=y_tr.to_numpy(),
            X_va=sc.transform(X_va[X_tr.columns].astype(float)).astype("float32"), y_va=y_va.to_numpy(),
            columns=list(X_tr.columns), scaler=sc,
        )
    return _DATA


# ======================================================================
# TensorFlow (필요할 때만)
# ======================================================================
_TF = {}


def tf_modules():
    if not _TF:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        tf.config.experimental.enable_op_determinism()   # 같은 PC·같은 시드면 같은 결과
        _TF.update(tf=tf, keras=keras, layers=layers)
    return _TF["tf"], _TF["keras"], _TF["layers"]


def make_optimizer(name, lr):
    _, keras, _ = tf_modules()
    if name == "SGD":   # sklearn 기본값과 같은 모멘텀 0.9 + 네스테로프
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name == "RMSprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    if name == "Adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "AdamW":
        return keras.optimizers.AdamW(learning_rate=lr)
    raise ValueError(name)


def build(cfg, input_dim):
    _, keras, layers = tf_modules()
    hidden = tuple(cfg["hidden"])
    seq = [layers.Input(shape=(input_dim,))]
    for i, units in enumerate(hidden):
        seq.append(layers.Dense(units, activation="relu", kernel_initializer="he_normal"))
        is_last = i == len(hidden) - 1
        where = cfg.get("drop_where", "none")
        if cfg["dropout"] > 0 and (where == "all" or (where == "last" and is_last)):
            seq.append(layers.Dropout(cfg["dropout"]))
    seq.append(layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(seq)
    model.compile(
        optimizer=make_optimizer(cfg["optimizer"], cfg["lr"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def _callbacks():
    _, keras, _ = tf_modules()

    class EpochSnapshot(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.weights, self.logs = [], []

        def on_epoch_end(self, epoch, logs=None):
            self.weights.append([w.copy() for w in self.model.get_weights()])
            self.logs.append({k: float(v) for k, v in (logs or {}).items()})

    class BestWeights(keras.callbacks.Callback):
        """끝까지 학습하고, 끝나면 검증 PR-AUC(Keras) 최고 에폭의 가중치로 되돌린다."""

        def on_train_begin(self, logs=None):
            self.best, self.weights = -np.inf, None

        def on_epoch_end(self, epoch, logs=None):
            v = (logs or {}).get("val_pr_auc")
            if v is not None and v > self.best:
                self.best, self.weights = v, self.model.get_weights()

        def on_train_end(self, logs=None):
            if self.weights is not None:
                self.model.set_weights(self.weights)

    return EpochSnapshot, BestWeights


# ======================================================================
# 학습 + 캐시
# ======================================================================
def class_weight_of(y, mode):
    if mode in (None, "none", "under_1to4"):
        return None
    if mode != "balanced":
        raise ValueError(f"지원하지 않는 class_weight: {mode}")
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


def under_1to4(x, y, seed):
    """학습 셋만 구매:비구매 = 1:4 로 언더샘플링(시드 고정)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keep = np.concatenate([pos, rng.choice(neg, 4 * len(pos), replace=False)])
    return x[keep], y[keep]


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


_CACHE = None


def read_cache(refresh=False):
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = pd.read_csv(RUNS_CSV) if os.path.isfile(RUNS_CSV) else pd.DataFrame(columns=EPOCH_COLS)
    return _CACHE


def cached_rows(cfg, seed):
    df = read_cache()
    if df.empty:
        return None
    hit = df[(df["cfg"] == key_of(cfg)) & (df["seed"] == seed)]
    return None if hit.empty else hit.sort_values("epoch")


class NotCached(Exception):
    pass


def run_one(cfg, seed):
    """한 시드 학습. 같은 설정+시드가 flow_runs.csv 에 있으면 학습하지 않는다."""
    cfg = cfg_of(**cfg)
    hit = cached_rows(cfg, seed)
    if hit is not None:
        return hit, "재사용"
    if CACHE_ONLY:
        raise NotCached(f"{key_of(cfg)} seed {seed}")

    _, keras, _ = tf_modules()
    EpochSnapshot, BestWeights = _callbacks()
    d = load_data()
    print(f"  [학습] {cfg_line(cfg)} · 시드 {seed}", flush=True)

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    x_fit, y_fit = d["X_tr"], d["y_tr"]
    if cfg["class_weight"] == "under_1to4":
        x_fit, y_fit = under_1to4(x_fit, y_fit, seed)
    cw = class_weight_of(y_fit, cfg["class_weight"])

    snap = EpochSnapshot()
    callbacks = [snap]
    if cfg["early_stop"]:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=cfg["early_stop"], restore_best_weights=True,
        ))
    else:
        callbacks.append(BestWeights())

    model = build(cfg, d["X_tr"].shape[1])
    t0 = time.perf_counter()
    history = model.fit(
        x_fit, y_fit,
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
    out.to_csv(RUNS_CSV, mode="a", header=not os.path.isfile(RUNS_CSV), index=False, encoding="utf-8-sig")
    read_cache(refresh=True)
    return out, "새로 실행"


def run_seeds(cfg, seeds=None):
    frames, flags = [], []
    for seed in (seeds or SEEDS):
        f, flag = run_one(cfg, seed)
        frames.append(f)
        flags.append(flag)
    status = "재사용" if all(x == "재사용" for x in flags) else "새로 실행"
    return pd.concat(frames, ignore_index=True), status


# ======================================================================
# 요약
# ======================================================================
def summarize(label, cfg, epoch_rows, status=""):
    """한 후보(3시드)의 요약 dict. 시드별 값은 'seeds' 에."""
    seeds = []
    for seed, sub in epoch_rows.groupby("seed"):
        sub = sub.sort_values("epoch")
        r = sub.iloc[0]
        best = int(r["best_epoch"])
        best_row = sub[sub["epoch"] == best].iloc[0]
        last = sub.iloc[-1]
        at30 = sub[sub["epoch"] == 30]
        seeds.append(dict(
            seed=int(seed),
            val_pr=float(r["report_val_pr"]), train_pr=float(r["report_train_pr"]),
            recall=float(r["report_recall"]), precision=float(r["report_precision"]),
            f1=float(r["report_f1"]), pos_rate=float(r["report_pos_rate"]),
            best_epoch=best, n_epochs=int(r["n_epochs"]),
            val_pr_30=float(at30["val_pr_sk"].iloc[0]) if len(at30) else float("nan"),
            train_pr_30=float(at30["train_pr"].iloc[0]) if len(at30) else float("nan"),
            val_pr_last=float(last["val_pr_sk"]),
            val_drop=float(best_row["val_pr_sk"] - last["val_pr_sk"]),
            train_rise=float(last["train_pr"] - best_row["train_pr"]),
            n_params=int(r["n_params"]), seconds=float(r["seconds"]),
        ))
    s = pd.DataFrame(seeds).sort_values("seed")
    v = s["val_pr"].to_numpy()
    return dict(
        label=label, cfg=cfg_of(**cfg), status=status, seeds=s,
        mean=float(v.mean()), std=float(v.std(ddof=1)),
        train=float(s["train_pr"].mean()),
        recall=float(s["recall"].mean()), precision=float(s["precision"].mean()),
        f1=float(s["f1"].mean()), pos_rate=float(s["pos_rate"].mean()),
        best_epoch=float(s["best_epoch"].mean()), n_epochs=float(s["n_epochs"].mean()),
        val_pr_30=float(s["val_pr_30"].mean()), val_pr_30_std=float(s["val_pr_30"].std(ddof=1)),
        val_pr_last=float(s["val_pr_last"].mean()),
        val_drop=float(s["val_drop"].mean()), train_rise=float(s["train_rise"].mean()),
        n_params=int(s["n_params"].iloc[0]), seconds=float(s["seconds"].mean()),
    )


def evaluate(candidates):
    """[(label, cfg), ...] -> 요약 리스트. 캐시에 없으면 학습."""
    out = []
    for label, cfg in candidates:
        rows, status = run_seeds(cfg)
        out.append(summarize(label, cfg, rows, status))
    return out


def curves(cfg):
    """에폭별 기록(3시드)."""
    rows, _ = run_seeds(cfg)
    return rows.sort_values(["seed", "epoch"])


# ======================================================================
# 판정
# ======================================================================
def pick_best(results):
    return max(results, key=lambda r: r["mean"])


def judge(results, current_label, rule):
    """
    rule = "best"  : 1위 선택
    rule = "keep"  : 1위와 현재 값의 차이가 두 후보 표준편차 중 큰 값 이내면 현재 값 유지
    반환: (선택된 결과, 판정 문장)
    """
    best = pick_best(results)
    cur = next(r for r in results if r["label"] == current_label)
    diff = best["mean"] - cur["mean"]
    bound = max(best["std"], cur["std"])
    if rule == "best":
        chosen = best
        why = "1위 선택 규칙"
    else:
        chosen = cur if diff <= bound else best
        why = "차이가 큰 표준편차 이내면 현재 값 유지" + (" → 유지" if chosen is cur else " → 변경")
    line = (f"판정: 1위 {best['label']}. 현재 값({cur['label']})과의 차이 {diff:+.4f}, "
            f"두 후보 표준편차 중 큰 값 {bound:.4f} → 선택 {chosen['label']} ({why})")
    return chosen, line


# ======================================================================
# 터미널 표
# ======================================================================
def disp_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def pad(text, width, align="left"):
    gap = width - disp_width(text)
    if gap <= 0:
        return text
    return (" " * gap + text) if align == "right" else (text + " " * gap)


COLUMNS = {
    "params": ("가중치 수", lambda r: f"{r['n_params']}"),
    "pr": ("PR-AUC", lambda r: f"{r['mean']:.4f}±{r['std']:.4f}"),
    "train": ("학습PR", lambda r: f"{r['train']:.3f}"),
    "gap": ("차이", lambda r: f"{r['train'] - r['mean']:+.3f}"),
    "recall": ("재현율", lambda r: f"{r['recall']:.3f}"),
    "precision": ("정밀도", lambda r: f"{r['precision']:.4f}"),
    "f1": ("F1", lambda r: f"{r['f1']:.3f}"),
    "pos": ("양성예측", lambda r: f"{r['pos_rate']:.3f}"),
    "best": ("최고/전체에폭", lambda r: f"{round(r['best_epoch'])}/{round(r['n_epochs'])}"),
    "val30": ("30에폭 검증PR", lambda r: f"{r['val_pr_30']:.4f}±{r['val_pr_30_std']:.4f}"),
    "drop": ("검증 하락", lambda r: f"{r['val_drop']:+.4f}"),
    "rise": ("학습 상승", lambda r: f"{r['train_rise']:+.3f}"),
    "sec": ("학습 시간", lambda r: f"{r['seconds']:.1f}"),
    "status": ("실행", lambda r: r["status"]),
}
DEFAULT_COLS = ["pr", "train", "gap", "recall", "precision", "pos", "best", "status"]


def table(results, cols=DEFAULT_COLS, first="후보"):
    heads = [first] + [COLUMNS[c][0] for c in cols]
    aligns = ["left"] + ["left" if c == "status" else "right" for c in cols]
    body = [[r["label"]] + [COLUMNS[c][1](r) for c in cols] for r in results]
    widths = [max(disp_width(row[i]) for row in [heads] + body) for i in range(len(heads))]
    return "\n".join("  ".join(pad(row[i], widths[i], aligns[i]) for i in range(len(row)))
                     for row in [heads] + body)


def seed_table(results):
    heads = ["후보", "시드", "검증PR", "학습PR", "차이", "최고에폭"]
    aligns = ["left"] + ["right"] * 5
    body = []
    for r in results:
        for _, s in r["seeds"].iterrows():
            body.append([r["label"], str(int(s["seed"])), f"{s['val_pr']:.4f}", f"{s['train_pr']:.3f}",
                         f"{s['train_pr'] - s['val_pr']:+.3f}", str(int(s["best_epoch"]))])
    widths = [max(disp_width(row[i]) for row in [heads] + body) for i in range(len(heads))]
    return "\n".join("  ".join(pad(row[i], widths[i], aligns[i]) for i in range(len(row)))
                     for row in [heads] + body)


def header(title, cfg_text):
    print(f"\n=== {title} ===")
    print(cfg_text)


def check_values(results, expected, tol=5e-5):
    """기존 결과와 같은지(평균 소수 4자리) 확인. expected = {label: mean}."""
    if not expected:
        return
    print("[확인] 기존 기록과 비교")
    for r in results:
        if r["label"] in expected:
            exp = expected[r["label"]]
            ok = abs(r["mean"] - exp) < tol
            print(f"  {r['label']}: 이번 {r['mean']:.4f}  기록 {exp:.4f}  {'일치' if ok else '불일치'}")
