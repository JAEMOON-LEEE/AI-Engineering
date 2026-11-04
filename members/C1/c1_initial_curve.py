"""
0단계 출발점의 에폭별 학습 곡선.

c1_ver2_adam.py의 전처리·모델·시드 설정을 그대로 가져온다.
캐시(ver2_adam_runs.csv)는 학습에 쓰지 않고, 30에폭 결과가 공식 숫자와 같은지
확인하는 데만 읽는다. 다르면 그래프 파일을 만들지 않는다.

SGD는 이 파일에서 다시 정의하지 않는다. 공식 make_optimizer가
학습률 5e-5, 모멘텀 0.9, 네스테로프 True 이다.
(논문 쪽 sklearn MLP 기본값. 모멘텀 0이 아니다.)

사용법 (아무 폴더에서):
  python members/C1/c1_initial_curve.py
  python members/C1/c1_initial_curve.py --epochs 300
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import c1_ver2_adam as v2
from c1_ver2_adam import keras, tf
from sklearn.metrics import average_precision_score, recall_score

C1_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(C1_DIR, "figures")
SEEDS = list(v2.SEEDS)
# 30에폭 공식 값과 이 정도까지 다르면 다른 실험으로 보고 그래프를 만들지 않는다.
MATCH_ATOL = 1e-4


class EpochSnapshot(keras.callbacks.Callback):
    """에폭이 끝날 때마다 가중치와 손실만 복사한다.

    여기서 predict()를 호출하면 전역 난수 상태가 바뀌어 다음 에폭의 배치 순서가
    공식 run_one()과 달라질 수 있다. PR-AUC는 학습이 끝난 뒤 복사해 둔 가중치로 계산한다.
    """

    def on_train_begin(self, logs=None):
        self.weights = []
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append([w.copy() for w in self.model.get_weights()])
        self.logs.append({k: float(v) for k, v in (logs or {}).items()})


def output_paths(epochs):
    stem = "initial_curve" if epochs == 30 else f"initial_curve_e{epochs}"
    return (
        os.path.join(FIG_DIR, stem + ".csv"),
        os.path.join(FIG_DIR, stem + ".png"),
    )


def train_one(cfg, seed, d):
    """c1_ver2_adam.run_one()과 같은 순서. 조기종료 없음, class_weight 없음."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    snap = EpochSnapshot()
    # 공식 스크립트도 조기종료가 없으면 BestWeights를 붙인다.
    # 이 콜백은 에폭 도중 가중치를 바꾸지 않고, 학습이 끝난 뒤에만 최고 에폭으로 되돌린다.
    # 곡선은 에폭마다 복사한 가중치로 계산하므로 그 복원과 무관하다.
    model = v2.build(cfg, d["X_tr"].shape[1])
    model.fit(
        d["X_tr"], d["y_tr"],
        validation_data=(d["X_va"], d["y_va"]),
        epochs=cfg["max_epochs"],
        batch_size=cfg["batch"],
        class_weight=None,
        callbacks=[v2.BestWeights(), snap],
        verbose=0,
    )
    rows = []
    for epoch, (weights, logs) in enumerate(zip(snap.weights, snap.logs), start=1):
        model.set_weights(weights)
        p_tr = model.predict(d["X_tr"], verbose=0).ravel()
        p_va = model.predict(d["X_va"], verbose=0).ravel()
        pred = (p_va >= 0.5).astype(int)
        rows.append(dict(
            seed=seed,
            epoch=epoch,
            train_pr=float(average_precision_score(d["y_tr"], p_tr)),
            val_pr=float(average_precision_score(d["y_va"], p_va)),
            val_recall=float(recall_score(d["y_va"], pred, zero_division=0)),
            val_pos_rate=float(pred.mean()),
            loss=float(logs["loss"]),
            val_loss=float(logs["val_loss"]),
        ))
    return rows


def load_official_initial():
    runs = pd.read_csv(v2.RUNS_CSV)
    key = v2.key_of(dict(v2.INITIAL))
    hit = runs[runs["cfg"] == key].drop_duplicates("seed")
    return hit, key


def compare_official(df, epochs):
    """30에폭 검증 PR-AUC가 공식 0단계 행과 같은지. 30이 아니면 비교하지 않는다."""
    if epochs != 30:
        print("[비교] --epochs 가 30이 아니라 공식 0단계(30에폭) 숫자와는 비교하지 않음")
        return True, None
    ref, key = load_official_initial()
    if set(ref["seed"].astype(int)) != set(SEEDS):
        print("[비교 실패] ver2_adam_runs.csv에서 0단계 시드 42/43/44 행을 찾지 못함")
        print("  key:", key)
        return False, ref
    last = df[df["epoch"] == epochs]
    print("\n[30에폭 검증 PR-AUC, sklearn average_precision_score]")
    print(f"{'seed':>6} {'이번':>12} {'공식':>12} {'차이':>12} {'학습PR':>12} {'공식학습PR':>12}")
    ok = True
    for seed in SEEDS:
        got = float(last.loc[last["seed"] == seed, "val_pr"].iloc[0])
        official = float(ref.loc[ref["seed"] == seed, "val_pr"].iloc[0])
        got_tr = float(last.loc[last["seed"] == seed, "train_pr"].iloc[0])
        official_tr = float(ref.loc[ref["seed"] == seed, "train_pr"].iloc[0])
        diff = got - official
        if abs(diff) > MATCH_ATOL or abs(got_tr - official_tr) > MATCH_ATOL:
            ok = False
        print(f"{seed:6d} {got:12.6f} {official:12.6f} {diff:12.6f} {got_tr:12.6f} {official_tr:12.6f}")
    mean = float(last["val_pr"].mean())
    std = float(last["val_pr"].std(ddof=1))
    ref_mean = float(ref["val_pr"].mean())
    ref_std = float(ref["val_pr"].std(ddof=1))
    train_mean = float(last["train_pr"].mean())
    print(f"\n이번 검증 PR-AUC {mean:.4f} ± {std:.4f}  (표본 표준편차)")
    print(f"공식 검증 PR-AUC {ref_mean:.4f} ± {ref_std:.4f}")
    print(f"30에폭 학습 PR-AUC 평균 {train_mean:.4f}")
    if abs(mean - ref_mean) > MATCH_ATOL or abs(std - ref_std) > MATCH_ATOL:
        ok = False
    return ok, ref


def best_epochs(df):
    print("\n[시드별 최고 검증 PR-AUC 에폭]")
    found = {}
    for seed, sub in df.groupby("seed"):
        i = sub["val_pr"].idxmax()
        row = sub.loc[i]
        found[int(seed)] = int(row["epoch"])
        print(f"  seed {int(seed)}: {int(row['epoch'])}에폭  {row['val_pr']:.6f}")
    return found


def plot_curve(df, epochs, png_path, val_rate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["axes.labelsize"] = 14
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["xtick.labelsize"] = 12
    matplotlib.rcParams["ytick.labelsize"] = 12
    matplotlib.rcParams["legend.fontsize"] = 12

    g = df.groupby("epoch")
    x = g.size().index.to_numpy()
    val_mean = g["val_pr"].mean()
    val_min = g["val_pr"].min()
    val_max = g["val_pr"].max()
    train_mean = g["train_pr"].mean()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(x, val_min, val_max, color="#1f77b4", alpha=0.18, label="검증 최소~최대")
    ax.plot(x, val_mean, color="#1f77b4", lw=2.4, label="검증 PR-AUC")
    ax.plot(x, train_mean, color="#ff7f0e", lw=2.0, ls="--", label="학습 PR-AUC")
    ax.axhline(val_rate, color="#555555", lw=1.4, ls="--", label="무작위 기준")
    ax.scatter([epochs], [float(val_mean.loc[epochs])], s=46, color="#1f77b4", zorder=5)
    ax.annotate(
        f"{epochs}에폭에서 학습 종료 (최고 = 마지막 에폭)",
        xy=(epochs, float(val_mean.loc[epochs])),
        xytext=(-18, 16),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="#333333"),
    )
    ax.set_xlim(1, epochs)
    ax.set_xlabel("에폭")
    ax.set_ylabel("PR-AUC")
    if epochs == 30:
        ax.set_title("출발점: 30에폭까지 계속 오르는 중")
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    else:
        ax.set_title(f"출발점: {epochs}에폭까지")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", color="#dddddd", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="0단계 초기값의 에폭별 PR-AUC 곡선")
    parser.add_argument("--epochs", type=int, default=30, help="최대 에폭 (기본 30)")
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs 는 1 이상이어야 합니다.")

    t0 = time.perf_counter()
    cfg = dict(v2.INITIAL)
    cfg["max_epochs"] = args.epochs
    d = v2.load_data()
    val_rate = float(np.mean(d["y_va"]))
    print(f"[env] tensorflow {tf.__version__}  keras {keras.__version__}")
    print(v2.describe(cfg))
    print("SGD는 공식 make_optimizer: 모멘텀 0.9, 네스테로프 True (모멘텀 없음이 아님)")
    print(f"[data] train={len(d['y_tr'])} val={len(d['y_va'])} 입력 {d['X_tr'].shape[1]}차원, "
          f"검증 구매율 {val_rate:.4f} (test 미사용)")

    rows = []
    for seed in SEEDS:
        t_seed = time.perf_counter()
        print(f"\n[학습] seed {seed}, {args.epochs}에폭")
        rows.extend(train_one(cfg, seed, d))
        last = rows[-1]
        print(f"  완료 ({time.perf_counter() - t_seed:.1f}초)  "
              f"마지막 에폭 검증 PR-AUC {last['val_pr']:.6f}  학습 PR-AUC {last['train_pr']:.6f}")
        if args.epochs == 30:
            ref, _key = load_official_initial()
            official = ref.loc[ref["seed"] == seed, "val_pr"]
            if len(official) and abs(float(official.iloc[0]) - last["val_pr"]) > 0.01:
                print(f"[중단] seed {seed} 검증 PR-AUC {last['val_pr']:.6f} "
                      f"vs 공식 {float(official.iloc[0]):.6f}. 나머지 시드는 돌리지 않습니다.")
                print(f"전체 {time.perf_counter() - t0:.1f}초")
                return 1

    df = pd.DataFrame(rows)
    peaks = best_epochs(df)
    ok, _ref = compare_official(df, args.epochs)
    still_rising = all(ep == args.epochs for ep in peaks.values())
    csv_path, png_path = output_paths(args.epochs)

    if args.epochs == 30 and not ok:
        print("\n[중단] 공식 0단계 숫자와 다릅니다. 그래프와 csv는 쓰지 않습니다.")
        print(f"전체 {time.perf_counter() - t0:.1f}초")
        return 1

    if args.epochs == 30 and not still_rising:
        print("\n[중단] 최고 검증 PR-AUC가 마지막 에폭이 아닌 시드가 있습니다. 그래프는 쓰지 않습니다.")
        print(f"전체 {time.perf_counter() - t0:.1f}초")
        return 1

    if not still_rising:
        print("\n[참고] 최고 검증 PR-AUC가 마지막 에폭이 아닌 시드가 있습니다. 측정값은 저장합니다.")

    os.makedirs(FIG_DIR, exist_ok=True)
    cols = ["seed", "epoch", "train_pr", "val_pr", "val_recall", "val_pos_rate", "loss", "val_loss"]
    df[cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    if args.epochs == 30:
        plot_curve(df, args.epochs, png_path, val_rate)
        print(f"\n저장: {csv_path}")
        print(f"      {png_path}")
    else:
        print(f"\n저장: {csv_path}")
    print(f"전체 {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
