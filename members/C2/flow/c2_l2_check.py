"""
과적합 대응 비교 · L2 규제 (교재 3.7.1 규제 강화)

출발은 c2_step05_overfit_fix.py 의 드롭아웃 비교와 같다.
3단계 결정: 28-56-28, ReLU + he_normal, Adam 0.01, 배치 128, 최대 30에폭,
클래스 가중치 balanced, 조기종료 없음, 드롭아웃 0.
은닉층 Dense 에만 kernel L2. 출력층은 제외. 시드 42, 43, 44.
평가: flow_core 그대로 (최고 에폭 가중치 → sklearn PR-AUC). test 는 읽지 않음.

후보: L2 0(캐시) / 1e-5 / 1e-4 / 1e-3 / 1e-2. 참고: 드롭아웃 0.2(캐시).
학습 전에 드롭아웃 0·0.2 캐시가 그대로인지 확인하고, 어긋나면 학습하지 않는다.

사용법: python members/C2/flow/c2_l2_check.py
"""
import os
import sys

C1_FLOW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "C1", "flow"))
if C1_FLOW not in sys.path:
    sys.path.insert(0, C1_FLOW)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import flow_core as fc  # noqa: E402

L2S = [
    (0.0, "L2 0"),
    (1e-5, "L2 1e-5"),
    (1e-4, "L2 1e-4"),
    (1e-3, "L2 1e-3"),
    (1e-2, "L2 1e-2"),
]
# 캐시에 들어 있는 cfg 문자열. l2 키를 기본값으로 넣으면 여기서 어긋난다.
KEY_DROP0 = (
    '{"batch": 128, "class_weight": "balanced", "drop_where": "none", "dropout": 0.0, '
    '"early_stop": 0, "hidden": [28, 56, 28], "lr": 0.01, "max_epochs": 30, "optimizer": "Adam"}'
)
KEY_DROP02 = (
    '{"batch": 128, "class_weight": "balanced", "drop_where": "all", "dropout": 0.2, '
    '"early_stop": 0, "hidden": [28, 56, 28], "lr": 0.01, "max_epochs": 30, "optimizer": "Adam"}'
)
CACHE_EXPECTED = {"드롭아웃 0": 0.6907, "드롭아웃 0.2": 0.6919}


def base_cfg():
    """5단계 드롭아웃 비교의 출발 = 3단계 결정."""
    return fc.cfg_of(
        hidden=(28, 56, 28),
        dropout=0.0,
        optimizer="Adam",
        lr=0.01,
        batch=128,
        max_epochs=30,
        early_stop=0,
        class_weight="balanced",
    )


def _mean_std(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def _pm(mean, std, sign=False):
    return f"{mean:+.4f}±{std:.4f}" if sign else f"{mean:.4f}±{std:.4f}"


def _from_cache(label, cfg):
    frames = []
    for seed in fc.SEEDS:
        hit = fc.cached_rows(cfg, seed)
        if hit is None:
            raise SystemExit(f"[중단] {label} 시드 {seed} 가 캐시에 없음\n{fc.key_of(cfg)}")
        frames.append(hit)
    rows = pd.concat(frames, ignore_index=True)
    return fc.summarize(label, cfg, rows, "재사용")


def confirm_cache(start):
    """드롭아웃 0 / 0.2 를 새로 학습하지 않고 캐시에서 읽는다. 어긋나면 종료."""
    drop0 = fc.cfg_of(**dict(start, l2=0))
    drop02 = fc.cfg_of(**dict(start, dropout=0.2, l2=0))
    print("[캐시 확인] 새로 학습하지 않음", flush=True)
    if "l2" in drop0 or "l2" in drop02:
        raise SystemExit("[중단] l2=0 인데 cfg 에 l2 키가 있음. 기존 캐시 키가 깨짐")
    if fc.key_of(drop0) != KEY_DROP0:
        raise SystemExit(f"[중단] 드롭아웃 0 캐시 키가 바뀜\n{fc.key_of(drop0)}")
    if fc.key_of(drop02) != KEY_DROP02:
        raise SystemExit(f"[중단] 드롭아웃 0.2 캐시 키가 바뀜\n{fc.key_of(drop02)}")
    got = {
        "드롭아웃 0": _from_cache("드롭아웃 0", drop0),
        "드롭아웃 0.2": _from_cache("드롭아웃 0.2", drop02),
    }
    for label, exp in CACHE_EXPECTED.items():
        mean = got[label]["mean"]
        ok = abs(mean - exp) < 5e-5
        print(f"  {label}: 이번 {mean:.4f}  기록 {exp:.4f}  {'일치' if ok else '불일치'}", flush=True)
        if not ok:
            raise SystemExit("[중단] 캐시 값이 기록과 다름. L2 학습을 시작하지 않음")
    return got


def confirm_regularizer():
    """은닉층에만 L2 가 붙고, l2=0 인 모델은 규제가 없는지 확인."""
    _, keras, _ = fc.tf_modules()

    def denses(cfg):
        model = fc.build(cfg, 8)
        return [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]

    off = denses(fc.cfg_of(l2=0))
    on = denses(fc.cfg_of(l2=1e-3))
    if len(off) != 4 or len(on) != 4:
        raise SystemExit(f"[중단] Dense 층 수 {len(off)}, {len(on)} (기대 4)")
    if any(layer.kernel_regularizer is not None for layer in off):
        raise SystemExit("[중단] l2=0 인데 규제가 붙어 있음")
    for layer in on[:-1]:
        reg = layer.kernel_regularizer
        coef = float(keras.backend.get_value(reg.l2)) if reg is not None else None
        if reg is None or abs(coef - 1e-3) > 1e-12:
            raise SystemExit(f"[중단] 은닉층 L2 계수 {coef}")
    if on[-1].kernel_regularizer is not None:
        raise SystemExit("[중단] 출력층에 L2 가 붙어 있음")
    keras.backend.clear_session()
    print("[모델 확인] 은닉층만 L2, 출력층 제외, l2=0 은 규제 없음", flush=True)


def _rows(results):
    body = []
    for r in results:
        s = r["seeds"].sort_values("seed")
        tr_m, tr_s = _mean_std(s["train_pr_30"])
        va_m, va_s = _mean_std(s["val_pr_30"])
        gap_m, gap_s = _mean_std(s["train_pr_30"].to_numpy() - s["val_pr_30"].to_numpy())
        bests = ", ".join(f"{int(row.seed)}:{int(row.best_epoch)}" for row in s.itertuples())
        body.append([
            r["label"],
            _pm(r["mean"], r["std"]),
            _pm(tr_m, tr_s),
            _pm(va_m, va_s),
            _pm(gap_m, gap_s, sign=True),
            bests,
            r["status"],
        ])
    return body


def show(results):
    heads = ["후보", "검증PR-AUC", "30에폭 학습PR", "30에폭 검증PR", "30에폭 차이", "최고에폭(시드별)", "실행"]
    aligns = ["left", "right", "right", "right", "right", "left", "left"]
    body = _rows(results)
    widths = [max(fc.disp_width(row[i]) for row in [heads] + body) for i in range(len(heads))]
    lines = [heads] + body
    return "\n".join(
        "  ".join(fc.pad(row[i], widths[i], aligns[i]) for i in range(len(row))) for row in lines
    )


def seed_detail(results):
    heads = ["후보", "시드", "검증PR", "최고에폭", "30학습PR", "30검증PR", "30차이"]
    aligns = ["left"] + ["right"] * 6
    body = []
    for r in results:
        for _, s in r["seeds"].sort_values("seed").iterrows():
            body.append([
                r["label"], str(int(s["seed"])), f"{s['val_pr']:.4f}", str(int(s["best_epoch"])),
                f"{s['train_pr_30']:.4f}", f"{s['val_pr_30']:.4f}",
                f"{s['train_pr_30'] - s['val_pr_30']:+.4f}",
            ])
    widths = [max(fc.disp_width(row[i]) for row in [heads] + body) for i in range(len(heads))]
    lines = [heads] + body
    return "\n".join(
        "  ".join(fc.pad(row[i], widths[i], aligns[i]) for i in range(len(row))) for row in lines
    )


def judgment(l2_results, ref):
    """L2 후보 중 1위와 드롭아웃 0.2 의 차이, 두 표준편차 중 큰 값."""
    best = fc.pick_best(l2_results)
    diff = best["mean"] - ref["mean"]
    bound = max(best["std"], ref["std"])
    line = (f"판정: 1위 {best['label']}. 현재 값({ref['label']})과의 차이 {diff:+.4f}, "
            f"두 후보 표준편차 중 큰 값 {bound:.4f} → 선택 {best['label']} (L2 후보 중 1위)")
    return line


def main():
    start = base_cfg()
    confirm_cache(start)
    confirm_regularizer()
    l2_cands = [(label, fc.cfg_of(**dict(start, l2=v))) for v, label in L2S]
    ref_cands = [("드롭아웃 0.2", fc.cfg_of(**dict(start, dropout=0.2)))]
    print("캐시 확인 통과. L2 그리드를 실행합니다.", flush=True)
    res = fc.evaluate(l2_cands)
    res_r = fc.evaluate(ref_cands)
    fc.header("L2 규제 비교 (C2)", fc.cfg_line(start))
    print("은닉층만 L2 (출력층 제외). 검증PR-AUC = 최고 에폭 가중치의 sklearn PR-AUC.")
    print("30에폭 차이 = 그 에폭의 학습PR − 검증PR. 최고에폭은 시드별(42, 43, 44).")
    print("[L2 후보]")
    print(show(res))
    print("[참고]")
    print(show(res_r))
    print(seed_detail(res + res_r))
    print(judgment(res, res_r[0]))
    fc.check_values(res + res_r, {"L2 0": 0.6907, "드롭아웃 0.2": 0.6919})
    return res, res_r


if __name__ == "__main__":
    main()
