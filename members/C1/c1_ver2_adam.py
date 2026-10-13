"""
C1 ver2 (Adam 고정판) - 증상 기반 하이퍼파라미터 결정 실험 (Keras)

[c1_ver2_diagnostic.py와 다른 점, 2026-10-13]
  1) 2단계 옵티마이저는 Adam으로 고정한다.
     16개 조합(SGD/RMSprop/Adam/AdamW × 학습률 4개)은 그대로 모두 돌려 표로 남기되,
     선택은 Adam 4개 중 검증 PR-AUC 평균 1위의 학습률로 한다.
     이유: 학습률 0.01에서 RMSprop·Adam·AdamW의 차이(0.004)가 시드 표준편차보다 작아 사실상 무승부이고,
           교재 5.4 [보충] "특별한 이유가 없다면 Adam(대규모 모델은 AdamW)부터 시작하라"를 따름.
  2) 5단계 배치 크기 비교는 증상 판정 없이 항상 실행하는 '확인' 단계로 바꾼다.
     (배치 크기와 시드 흔들림을 잇는 가설은 교재 근거가 약해서 뺌. 교재 4.5.2 범위 32~256에서 비교 후 1위 선택)
  3) 결과는 ver2_adam_runs.csv / ver2_adam_steps.csv / ver2_adam_story.md에 따로 저장한다.
     (기존 ver2_runs.csv는 클라우드 환경에서 돈 결과라 읽지 않는다. 환경이 섞이면 같은 시드라도 숫자가 달라
      단계 안 비교가 공정하지 않음 -> 모든 실행을 이 PC에서 새로 돌린다.)
  4) TensorFlow 결정론 모드를 켠다. 같은 PC에서 다시 돌리면 같은 숫자가 나온다.
  5) 7단계 '구조 넓게 확인'을 추가한다. 최종 학습 조건에서 1~3층 × 32/64/128 노드(9개)를 현재 구조와 비교.
     1위와 현재 구조의 차이가 표준편차 이내면 현재 구조 유지(실행 전에 정한 규칙).
  6) 8단계 '에폭 재확인'을 추가한다. 최종 조건에서 최대 30/50/100에폭(최고 에폭 가중치)을 비교. 규칙은 7단계와 같음.

원칙: "증상 -> 원인 가설 -> 실험 -> 결정"
  (수정 2026-10-12) 모든 후보는 '검증 PR-AUC가 가장 좋았던 에폭'의 가중치로 평가한다.
  조기종료가 없는 설정도 최대 에폭까지 학습한 뒤 최고 에폭으로 되돌려 비교한다.
  -> 학습 속도가 다른 설정끼리 공정하게 비교하기 위해. 조기종료 단계는 '언제 멈출지(시간·에폭 수)'를 정하는 의미가 됨.
  직전 설정의 실험 결과에서 정해 둔 증상이 보일 때만 그 증상과 관련된 설정을 바꾼다.
  증상이 없으면 바꾸지 않는다(= 초기값을 유지할 실험 근거).
  후보 중에서는 검증 PR-AUC 평균이 가장 높은 것을 고른다.

초기값 (논문에 있으면 논문, 없으면 교재)
  은닉층 (28, 56, 28), ReLU ............ Tanvir et al. 2023, 3.7절 iii항 (교재 5.5.1도 은닉층 ReLU)
  옵티마이저 SGD, 학습률 0.00005 ........ 같은 곳. 논문은 scikit-learn으로 구현(4장)했으므로
                                         sklearn 기본값인 모멘텀 0.9 + 네스테로프를 그대로 옮김
  불균형 처리 없음 ...................... 논문 기본 실험(Table 5)
  배치 128, 최대 30에폭 ................. 논문에 없음 -> 교재 5.4.1 [참고] 코드
  출력 sigmoid + 이진 교차 엔트로피 ....... 교재 5.4.1 (교차 엔트로피 권장)
  he_normal 초기화 ...................... 교재 5.5.4 (ReLU에는 He 초기화)
  조기종료·드롭아웃 없음 ................. 논문에 없음, 교재 5.1.1은 기법만 소개

증상 판정 기준 (실험 전에 고정)
  [불균형]   재현율 < 0.30  또는  양성 예측 비율 < 실제 구매율(학습 셋)의 절반
  [느린 학습] 최고 에폭이 최대 에폭의 80% 이후 (마지막까지 검증 PR-AUC가 오름)
  [에폭]     최고 에폭이 최대 에폭의 80% 이후 -> 에폭 부족 / 최고 에폭 이후 검증 손실이 최저점보다 10% 이상 상승 -> 과잉 적합
  [과잉 적합] 학습 PR-AUC - 검증 PR-AUC >= 0.05
  [과소 적합] 학습 PR-AUC - 검증 PR-AUC <  0.02
  [불안정]   시드 간 검증 PR-AUC 표준편차 >= 0.010

진행 순서 (앞 단계가 안 고쳐지면 뒤 단계 비교가 무의미한 순서)
  1 불균형 -> 2 느린 학습(옵티마이저×학습률) -> 3 에폭·조기종료 -> 4 모델 크기 -> 5 불안정(배치) -> 6 학습률 재확인
  6은 증상 판정이 아니라 단계 순서 때문에 생길 수 있는 편향을 확인하는 단계(항상 실행).

데이터: data/final/train_clean.csv -> validation_final.csv, 기존 C1과 같은 전처리(26차원). test는 읽지 않음.
시드 42/43/44.

사용법: python c1_ver2_diagnostic.py      (실행 결과는 캐시돼서 다시 실행하면 이어서 진행)
출력:  ver2_runs.csv(실행별 지표), ver2_steps.csv(단계별 후보 요약), ver2_story.md(발표용 흐름)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

tf.config.experimental.enable_op_determinism()   # 같은 PC·같은 시드면 항상 같은 결과

C1_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(C1_DIR, "..", "..", "data", "final")
RUNS_CSV = os.path.join(C1_DIR, "ver2_adam_runs.csv")
STEPS_CSV = os.path.join(C1_DIR, "ver2_adam_steps.csv")
STORY_MD = os.path.join(C1_DIR, "ver2_adam_story.md")
SEEDS = [42, 43, 44]
FIXED_OPTIMIZER = "Adam"   # 2단계에서 고를 옵티마이저 (교재 5.4 [보충] 권고)

INITIAL = dict(
    hidden=(28, 56, 28), dropout=0.0, optimizer="SGD", lr=5e-5,
    imbalance="none", batch=128, max_epochs=30, early_stop=0,
)
SOURCES = {
    "hidden": "논문 3.7절", "optimizer": "논문 3.7절", "lr": "논문 3.7절",
    "imbalance": "논문 Table 5", "batch": "교재 5.4.1", "max_epochs": "교재 5.4.1",
    "dropout": "논문·교재에 값 없음", "early_stop": "논문에 없음",
}

# 증상 기준
RECALL_MIN = 0.30
LATE_BEST = 0.80
VAL_LOSS_RISE = 0.10
GAP_OVER = 0.05
GAP_UNDER = 0.02
STD_UNSTABLE = 0.010


# ---------------- 데이터 ----------------
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


def load_data():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train_clean.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "validation_final.csv"))
    X_tr, y_tr = preprocess(tr)
    X_va, y_va = preprocess(va, ref_columns=X_tr.columns)
    sc = StandardScaler()
    return dict(
        X_tr=sc.fit_transform(X_tr.astype(float)).astype("float32"), y_tr=y_tr.to_numpy(),
        X_va=sc.transform(X_va[X_tr.columns].astype(float)).astype("float32"), y_va=y_va.to_numpy(),
    )


# ---------------- 모델 ----------------
def make_optimizer(name, lr):
    if name == "SGD":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name == "RMSprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    if name == "Adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "AdamW":
        return keras.optimizers.AdamW(learning_rate=lr)
    raise ValueError(name)


def build(cfg, input_dim):
    seq = [layers.Input(shape=(input_dim,))]
    for u in cfg["hidden"]:
        seq.append(layers.Dense(u, activation="relu", kernel_initializer="he_normal"))
        if cfg["dropout"] > 0:
            seq.append(layers.Dropout(cfg["dropout"]))
    seq.append(layers.Dense(1, activation="sigmoid"))
    m = keras.Sequential(seq)
    m.compile(optimizer=make_optimizer(cfg["optimizer"], cfg["lr"]), loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
    return m


class BestWeights(keras.callbacks.Callback):
    """학습은 최대 에폭까지 끝까지 하되, 끝나면 검증 PR-AUC 최고 에폭의 가중치로 되돌린다."""

    def on_train_begin(self, logs=None):
        self.best, self.weights = -np.inf, None

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get("val_pr_auc")
        if v is not None and v > self.best:
            self.best, self.weights = v, self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.weights is not None:
            self.model.set_weights(self.weights)


def run_one(cfg, seed, d):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    X, y = d["X_tr"], d["y_tr"]
    cw = None
    if cfg["imbalance"] == "class_weight":
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        cw = {0: float(w[0]), 1: float(w[1])}
    elif cfg["imbalance"] == "under_1to4":
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        keep = np.concatenate([pos, rng.choice(neg, 4 * len(pos), replace=False)])
        X, y = X[keep], y[keep]
    cbs = []
    if cfg["early_stop"]:
        cbs.append(keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                                 patience=cfg["early_stop"], restore_best_weights=True))
    else:
        # 조기종료가 없어도 '마지막 에폭'이 아니라 '검증 PR-AUC가 가장 좋았던 에폭'의 가중치로 평가한다.
        # (ver2 첫 실행에서는 마지막 에폭으로 비교해, 빨리 배우는 설정이 끝에 과잉 적합된 상태로 불리하게 비교됐음)
        cbs.append(BestWeights())
    m = build(cfg, X.shape[1])
    t0 = time.perf_counter()
    h = m.fit(X, y, validation_data=(d["X_va"], d["y_va"]), epochs=cfg["max_epochs"],
              batch_size=cfg["batch"], class_weight=cw, callbacks=cbs, verbose=0).history
    secs = time.perf_counter() - t0

    val_pr_curve = h["val_pr_auc"]
    val_loss = h["val_loss"]
    best_ep = int(np.argmax(val_pr_curve)) + 1
    p_va = m.predict(d["X_va"], verbose=0).ravel()
    p_tr = m.predict(d["X_tr"], verbose=0).ravel()   # 과잉 적합 진단은 원래 학습 셋 전체로
    pred = (p_va >= 0.5).astype(int)
    min_loss = float(np.min(val_loss))
    return dict(
        seed=seed,
        val_pr=average_precision_score(d["y_va"], p_va),
        train_pr=average_precision_score(d["y_tr"], p_tr),
        recall=recall_score(d["y_va"], pred, zero_division=0),
        precision=precision_score(d["y_va"], pred, zero_division=0),
        f1=f1_score(d["y_va"], pred, zero_division=0),
        pos_rate=float(pred.mean()),
        epochs=len(val_pr_curve), best_epoch=best_ep,
        val_loss_rise=(float(val_loss[-1]) - min_loss) / min_loss,
        n_params=int(m.count_params()), seconds=round(secs, 1),
        val_pr_curve=json.dumps([round(float(v), 4) for v in val_pr_curve]),
    )


# ---------------- 캐시·요약 ----------------
def key_of(cfg):
    return json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(cfg.items())})


_cache = None


def evaluate(cfg, d):
    """cfg를 시드 3개로 돌려 요약을 돌려준다. 이미 돈 조합은 ver2_runs.csv에서 읽는다."""
    global _cache
    if _cache is None:
        _cache = pd.read_csv(RUNS_CSV) if os.path.exists(RUNS_CSV) else pd.DataFrame(columns=["cfg", "seed"])
    k = key_of(cfg)
    rows = []
    for s in SEEDS:
        hit = _cache[(_cache["cfg"] == k) & (_cache["seed"] == s)]
        if len(hit):
            rows.append(hit.iloc[0].to_dict())
            continue
        r = run_one(cfg, s, d)
        r["cfg"] = k
        pd.DataFrame([r]).to_csv(RUNS_CSV, mode="a", header=not os.path.exists(RUNS_CSV),
                                 index=False, encoding="utf-8-sig")
        _cache = pd.concat([_cache, pd.DataFrame([r])], ignore_index=True)
        rows.append(r)
    df = pd.DataFrame(rows)
    return dict(
        val_pr=df["val_pr"].mean(), val_pr_std=df["val_pr"].std(),
        train_pr=df["train_pr"].mean(), gap=(df["train_pr"] - df["val_pr"]).mean(),
        recall=df["recall"].mean(), f1=df["f1"].mean(), pos_rate=df["pos_rate"].mean(),
        epochs=df["epochs"].mean(), best_epoch=df["best_epoch"].mean(),
        late_best=bool(((df["best_epoch"] / cfg["max_epochs"]) >= LATE_BEST).mean() >= 0.5),
        loss_rise=df["val_loss_rise"].mean(), n_params=int(df["n_params"].iloc[0]),
    )


def describe(cfg):
    h = "(" + ",".join(map(str, cfg["hidden"])) + ")"
    es = f"조기종료 p{cfg['early_stop']}" if cfg["early_stop"] else "조기종료 없음"
    imb = {"none": "불균형 처리 없음", "class_weight": "클래스 가중치", "under_1to4": "언더샘플링 1:4"}[cfg["imbalance"]]
    dr = f", 드롭아웃 {cfg['dropout']}" if cfg["dropout"] else ""
    return f"{h}{dr}, {cfg['optimizer']} {cfg['lr']:g}, 배치 {cfg['batch']}, 최대 {cfg['max_epochs']}에폭, {es}, {imb}"


def line(tag, s):
    return (f"{tag:<34} PR-AUC {s['val_pr']:.4f}±{s['val_pr_std']:.4f}  학습PR {s['train_pr']:.3f}  "
            f"차이 {s['gap']:+.3f}  재현율 {s['recall']:.3f}  양성예측 {s['pos_rate']:.3f}  "
            f"최고/전체에폭 {s['best_epoch']:.0f}/{s['epochs']:.0f}")


# ---------------- 단계 진행 ----------------
def main():
    d = load_data()
    base_rate = float(d["y_tr"].mean())
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} 입력 {d['X_tr'].shape[1]}차원, "
          f"학습 셋 구매율 {base_rate:.3f} (test 미사용)\n")

    cfg = dict(INITIAL)
    step_rows, story = [], []

    def try_candidates(step, symptom, hypothesis, cands, choose_from=None):
        """choose_from: 선택 대상을 좁히는 함수(cfg -> bool). 없으면 전체 후보 중 1위."""
        nonlocal cfg
        print(f"  가설: {hypothesis}")
        results = []
        for label, c in cands:
            s = evaluate(c, d)
            results.append((label, c, s))
            print("   " + line(label, s))
            step_rows.append(dict(step=step, symptom=symptom, candidate=label, config=describe(c),
                                  **{k: v for k, v in s.items()}))
        pool = [t for t in results if choose_from is None or choose_from(t[1])]
        best_label, best_cfg, best_s = max(pool, key=lambda t: t[2]["val_pr"])
        overall = max(results, key=lambda t: t[2]["val_pr"])
        if overall[0] != best_label:
            print(f"  (참고: 전체 1위는 {overall[0]} {overall[2]['val_pr']:.4f}, "
                  f"선택한 {best_label}과 차이 {overall[2]['val_pr'] - best_s['val_pr']:+.4f})")
        for r in step_rows[-len(results):]:
            r["selected"] = r["candidate"] == best_label
        changed = key_of(best_cfg) != key_of(cfg)
        print(f"  -> 선택: {best_label}" + ("" if changed else " (기존 설정 유지)"))
        story.append(f"- **{step}**: 증상 [{symptom}] → 가설: {hypothesis} → 후보 {len(results)}개 비교 → "
                     f"**{best_label}** 선택 (검증 PR-AUC {best_s['val_pr']:.4f}±{best_s['val_pr_std']:.4f}, "
                     f"재현율 {best_s['recall']:.3f})")
        cfg = best_cfg
        return best_s

    # 0. 출발점
    print("=== 0 출발점: 논문 + 교재 초기값 ===")
    print("  " + describe(cfg))
    cur = evaluate(cfg, d)
    print("   " + line("초기값", cur))
    step_rows.append(dict(step="0 출발점", symptom="-", candidate="초기값", config=describe(cfg), selected=True, **cur))
    story.append(f"- **0 출발점** (논문+교재 초기값: {describe(cfg)}): 검증 PR-AUC {cur['val_pr']:.4f}, "
                 f"재현율 {cur['recall']:.3f}, 양성 예측 비율 {cur['pos_rate']:.3f} (실제 구매율 {base_rate:.3f})")

    # 1. 불균형
    print("\n=== 1 불균형 ===")
    sym = cur["recall"] < RECALL_MIN or cur["pos_rate"] < base_rate / 2
    desc = f"재현율 {cur['recall']:.3f}, 양성 예측 {cur['pos_rate']:.3f} vs 실제 {base_rate:.3f}"
    if sym:
        print(f"  증상 있음: {desc}")
        cur = try_candidates("1 불균형", desc, "소수 클래스(구매)를 무시하도록 학습됨",
                             [(n, dict(cfg, imbalance=v)) for n, v in
                              [("처리 없음", "none"), ("클래스 가중치", "class_weight"), ("언더샘플링 1:4", "under_1to4")]])
    else:
        print(f"  증상 없음 ({desc}) -> 유지")
        story.append(f"- **1 불균형**: 증상 없음 ({desc}) → 유지")

    # 2. 느린 학습
    print("\n=== 2 느린 학습 ===")
    desc = f"최고 에폭 {cur['best_epoch']:.0f}/{cfg['max_epochs']} (마지막까지 개선 중: {cur['late_best']})"
    if cur["late_best"]:
        print(f"  증상 있음: {desc}")
        cands = [(f"{o} {lr:g}", dict(cfg, optimizer=o, lr=lr))
                 for o in ["SGD", "RMSprop", "Adam", "AdamW"] for lr in [5e-5, 1e-4, 1e-3, 1e-2]]
        cur = try_candidates("2 느린 학습", desc, "학습률이 작거나 옵티마이저가 느려 정해진 에폭 안에 수렴하지 못함",
                             cands, choose_from=lambda c: c["optimizer"] == FIXED_OPTIMIZER)
        story.append(f"  - 옵티마이저는 {FIXED_OPTIMIZER}로 고정: 적응형 옵티마이저끼리는 차이가 표준편차보다 작아 무승부로 보고, "
                     f"교재 5.4 [보충] 권고(특별한 이유가 없으면 Adam)를 따름. 학습률은 {FIXED_OPTIMIZER} 후보 중 1위")
    else:
        print(f"  증상 없음 ({desc}) -> 유지")
        story.append(f"- **2 느린 학습**: 증상 없음 ({desc}) → 옵티마이저·학습률 유지")

    # 3. 에폭·조기종료
    print("\n=== 3 에폭·조기종료 ===")
    too_short = cur["late_best"]
    overfit_time = cur["loss_rise"] >= VAL_LOSS_RISE
    desc = (f"최고 에폭 {cur['best_epoch']:.0f}/{cfg['max_epochs']}, "
            f"최저점 대비 마지막 검증 손실 {cur['loss_rise']:+.0%}")
    if too_short or overfit_time:
        kind = "에폭 부족" if too_short else "최고점 이후 과잉 적합"
        print(f"  증상 있음({kind}): {desc}")
        cands = [(f"최대 {cfg['max_epochs']}에폭, 조기종료 없음", dict(cfg))]
        if too_short:
            cands.append(("최대 100에폭, 조기종료 없음", dict(cfg, max_epochs=100, early_stop=0)))
        cands += [("최대 100에폭 + 조기종료 p5", dict(cfg, max_epochs=100, early_stop=5)),
                  ("최대 100에폭 + 조기종료 p10", dict(cfg, max_epochs=100, early_stop=10))]
        cur = try_candidates("3 에폭·조기종료", desc, kind, cands)
    else:
        print(f"  증상 없음 ({desc}) -> 유지")
        story.append(f"- **3 에폭·조기종료**: 증상 없음 ({desc}) → 유지")

    # 4. 모델 크기
    print("\n=== 4 모델 크기 ===")
    h = cfg["hidden"]
    half = tuple(max(1, u // 2) for u in h)
    double = tuple(u * 2 for u in h)
    desc = f"학습 PR {cur['train_pr']:.3f} - 검증 PR {cur['val_pr']:.3f} = {cur['gap']:+.3f}"
    if cur["gap"] >= GAP_OVER:
        print(f"  증상 있음(과잉 적합): {desc}")
        cands = [("현재 구조", dict(cfg)),
                 (f"절반 구조 {half}", dict(cfg, hidden=half)),
                 (f"얕은 구조 {h[:2]}", dict(cfg, hidden=h[:2])),
                 ("현재 구조 + 드롭아웃 0.2", dict(cfg, dropout=0.2)),
                 ("현재 구조 + 드롭아웃 0.3", dict(cfg, dropout=0.3))]
        cur = try_candidates("4 모델 크기", desc, "모델이 학습 데이터에 비해 커서 과잉 적합", cands)
    elif cur["gap"] < GAP_UNDER:
        print(f"  증상 있음(과소 적합): {desc}")
        cands = [("현재 구조", dict(cfg)),
                 (f"두 배 구조 {double}", dict(cfg, hidden=double)),
                 (f"더 깊은 구조 {h + (h[-1],)}", dict(cfg, hidden=h + (h[-1],)))]
        cur = try_candidates("4 모델 크기", desc, "모델이 작아 패턴을 다 담지 못함", cands)
    else:
        print(f"  증상 없음 ({desc}) -> 유지")
        story.append(f"- **4 모델 크기**: 증상 없음 ({desc}) → 논문 구조 유지")

    # 5. 배치 크기 확인 (증상 판정 없이 항상 실행)
    print("\n=== 5 배치 크기 확인 ===")
    desc = f"확인 단계(증상 판정 아님), 현재 시드 간 표준편차 {cur['val_pr_std']:.4f}"
    cands = [(f"배치 {b}", dict(cfg, batch=b)) for b in sorted({32, 64, 128, 256, cfg["batch"]})]
    cur = try_candidates("5 배치 크기 확인", desc, "교재 범위(32~256, 4.5.2)에서 비교해 1위 선택", cands)

    # 6. 학습률 재확인 (단계 순서 편향 확인, 항상 실행)
    print("\n=== 6 학습률 재확인 ===")
    lr0 = cfg["lr"]
    cands = [(f"학습률 {v:g}", dict(cfg, lr=v)) for v in sorted({lr0 / 3, lr0, lr0 * 3})]
    cur = try_candidates("6 학습률 재확인", "단계 순서 편향 확인(증상 판정 아님)",
                         "앞 단계에서 다른 조건일 때 고른 학습률이 최종 조건에서도 맞는지", cands)

    # 7. 구조 넓게 확인 (항상 실행, 2026-10-13 추가)
    #   4단계는 과잉 적합 증상 때문에 '줄인 구조'만 비교했으므로, 최종 학습 조건에서
    #   층 수 1~3 × 노드 32/64/128 (9개)과 현재(논문) 구조를 함께 비교한다.
    #   교재 4.10.1 그리드 서치 방식. 범위는 논문 구조(28·56 노드)를 포함하도록 잡음.
    #   [결과 보기 전에 정한 규칙] 1위 구조와 현재 구조의 평균 차이가 두 후보 표준편차 중 큰 값보다
    #   작거나 같으면 현재 구조를 유지한다(무승부). 클 때만 1위 구조로 바꾼다.
    print("\n=== 7 구조 넓게 확인 ===")
    base_h = cfg["hidden"]
    cands = [(f"현재 구조 {base_h}", dict(cfg))]
    for n_layers in (1, 2, 3):
        for u in (32, 64, 128):
            h = tuple([u] * n_layers)
            if h != base_h:
                cands.append((f"{n_layers}층×{u} {h}", dict(cfg, hidden=h)))
    print("  가설: 줄인 구조만 본 4단계보다 넓은 범위에서도 현재 구조가 맞는지")
    results = []
    for label, c in cands:
        s = evaluate(c, d)
        results.append((label, c, s))
        print("   " + line(label, s))
        step_rows.append(dict(step="7 구조 넓게 확인", symptom="확인 단계(증상 판정 아님)", candidate=label,
                              config=describe(c), **s))
    base_label, base_cfg, base_s = results[0]
    top_label, top_cfg, top_s = max(results, key=lambda t: t[2]["val_pr"])
    diff = top_s["val_pr"] - base_s["val_pr"]
    tol = max(top_s["val_pr_std"], base_s["val_pr_std"])
    if top_label != base_label and diff > tol:
        chosen = (top_label, top_cfg, top_s)
        verdict = f"1위 {top_label}이 현재 구조보다 {diff:+.4f} 높고 표준편차({tol:.4f})보다 커서 변경"
    else:
        chosen = (base_label, base_cfg, base_s)
        verdict = (f"1위 {top_label} (차이 {diff:+.4f})가 표준편차({tol:.4f}) 이내라 무승부 -> 현재 구조 유지"
                   if top_label != base_label else "현재 구조가 1위 -> 유지")
    for r in step_rows[-len(results):]:
        r["selected"] = r["candidate"] == chosen[0]
    print(f"  -> {verdict}")
    if chosen[0] != base_label:
        print("  ※ 구조가 바뀌었으므로 배치·학습률을 이 구조에서 다시 확인하는 게 좋음 (자동 재실행은 안 함)")
    story.append(f"- **7 구조 넓게 확인**: 1~3층 × 32/64/128 노드 9개 + 현재 구조 비교 → {verdict}")
    cfg, cur = chosen[1], chosen[2]

    # 8. 에폭 재확인 (항상 실행, 2026-10-13 추가)
    #   3단계는 드롭아웃·학습률이 바뀌기 전 조건에서 했으므로, 최종 조건에서 최대 에폭을 다시 확인한다.
    #   모두 '검증 최고 에폭 가중치'로 평가. 규칙은 7단계와 같음(표준편차 이내면 현재 값 유지).
    print("\n=== 8 에폭 재확인 ===")
    print(f"  현재 최고/최대 에폭 {cur['best_epoch']:.0f}/{cfg['max_epochs']}")
    print("  가설: 드롭아웃·작은 학습률로 최고점이 늦어져 현재 최대 에폭이 부족할 수 있음")
    ep0 = cfg["max_epochs"]
    cands = [(f"최대 {e}에폭", dict(cfg, max_epochs=e)) for e in sorted({ep0, 50, 100})]
    results = []
    for label, c in cands:
        s = evaluate(c, d)
        results.append((label, c, s))
        print("   " + line(label, s))
        step_rows.append(dict(step="8 에폭 재확인", symptom=f"최고 에폭 {cur['best_epoch']:.0f}/{ep0}", candidate=label,
                              config=describe(c), **s))
    base_r = next(t for t in results if t[1]["max_epochs"] == ep0)
    top_r = max(results, key=lambda t: t[2]["val_pr"])
    diff = top_r[2]["val_pr"] - base_r[2]["val_pr"]
    tol = max(top_r[2]["val_pr_std"], base_r[2]["val_pr_std"])
    if top_r[0] != base_r[0] and diff > tol:
        chosen = top_r
        verdict = f"{top_r[0]}이 {diff:+.4f} 높고 표준편차({tol:.4f})보다 커서 변경"
    else:
        chosen = base_r
        verdict = (f"1위 {top_r[0]} (차이 {diff:+.4f})가 표준편차({tol:.4f}) 이내라 무승부 -> 최대 {ep0}에폭 유지"
                   if top_r[0] != base_r[0] else f"최대 {ep0}에폭이 1위 -> 유지")
    for r in step_rows[-len(results):]:
        r["selected"] = r["candidate"] == chosen[0]
    print(f"  -> {verdict}")
    story.append(f"- **8 에폭 재확인**: 최종 조건에서 최대 {', '.join(str(c[1]['max_epochs']) for c in cands)}에폭 비교 → {verdict}")
    cfg, cur = chosen[1], chosen[2]

    # 결과 저장
    pd.DataFrame(step_rows).to_csv(STEPS_CSV, index=False, encoding="utf-8-sig")
    init_s = evaluate(dict(INITIAL), d)
    final_txt = describe(cfg)
    md = ["# C1 ver2 (Adam 고정판) - 증상 기반 하이퍼파라미터 결정 흐름", "",
          "검증 세트(validation_final) 기준, 시드 42/43/44 평균±표준편차. test는 사용하지 않음.", "",
          "## 초기값 출처",
          *[f"- {k}: {INITIAL[k]} ({SOURCES[k]})" for k in INITIAL], "",
          "## 단계별 흐름", *story, "",
          "## 결과",
          f"- 초기값: 검증 PR-AUC {init_s['val_pr']:.4f}, 재현율 {init_s['recall']:.3f}",
          f"- 최종: {final_txt}",
          f"- 최종 검증 PR-AUC {cur['val_pr']:.4f}±{cur['val_pr_std']:.4f}, 재현율 {cur['recall']:.3f}, F1 {cur['f1']:.3f}",
          "", "## 바뀌지 않은 항목 (초기값 유지: 증상이 없었거나, 증상은 있었지만 후보가 더 낫지 않았음 -> 위 흐름 참고)",
          *[f"- {k}: {INITIAL[k]} ({SOURCES[k]})" for k in INITIAL if INITIAL[k] == cfg[k]]]
    with open(STORY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n최종 설정: " + final_txt)
    print(f"최종 검증 PR-AUC {cur['val_pr']:.4f}±{cur['val_pr_std']:.4f}")
    print(f"저장: {RUNS_CSV}\n      {STEPS_CSV}\n      {STORY_MD}")


if __name__ == "__main__":
    main()
