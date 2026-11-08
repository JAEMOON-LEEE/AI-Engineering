"""
과적합 대응 기법 비교 출력. 캐시만 읽고 학습하지 않는다.

후보: 없음(출발), 드롭아웃 0.2, L2 1e-4, 단순화 (14,28,14). 조기종료는 제외.
출발 조건은 step04_overfit.compute() 결과. 시드 42/43/44.

사용법: python members/C2/flow/c2_step05b_print.py
"""
import os
import sys

os.environ["FLOW_CACHE_ONLY"] = "1"

C1_FLOW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "C1", "flow"))
if C1_FLOW not in sys.path:
    sys.path.insert(0, C1_FLOW)

import pandas as pd  # noqa: E402

import flow_core as fc  # noqa: E402
from step04_overfit import compute as prev  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C2_CSV = os.path.join(HERE, "c2_flow_runs.csv")


def _c2_hit(cfg, seed):
    if not os.path.isfile(C2_CSV):
        return None
    df = pd.read_csv(C2_CSV)
    if df.empty:
        return None
    hit = df[(df["cfg"] == fc.key_of(cfg)) & (df["seed"] == seed)]
    return None if hit.empty else hit.sort_values("epoch")


def rows_of(cfg):
    """c2_flow_runs.csv 다음 flow_runs.csv. 없으면 중단(학습하지 않음)."""
    cfg = fc.cfg_of(**cfg)
    frames = []
    for seed in fc.SEEDS:
        hit = _c2_hit(cfg, seed)
        if hit is None:
            hit = fc.cached_rows(cfg, seed)
        if hit is None:
            raise SystemExit(f"[중단] 캐시 없음: {fc.cfg_line(cfg)} 시드 {seed}")
        frames.append(hit)
    return pd.concat(frames, ignore_index=True)


def summarize(label, rows):
    seeds = []
    for _, sub in rows.groupby("seed"):
        sub = sub.sort_values("epoch")
        best = int(sub.iloc[0]["best_epoch"])
        best_val = float(sub.loc[sub["epoch"] == best, "val_pr_sk"].iloc[0])
        at30 = sub.loc[sub["epoch"] == 30, "val_pr_sk"]
        if at30.empty:
            raise SystemExit(f"[중단] {label} 시드 {int(sub.iloc[0]['seed'])} 에 30에폭 기록이 없음")
        val30 = float(at30.iloc[0])
        seeds.append(dict(
            val_pr=float(sub.iloc[0]["report_val_pr"]),
            best_epoch=best,
            val30=val30,
            drop=best_val - val30,
        ))
    frame = pd.DataFrame(seeds)
    pr = frame["val_pr"].to_numpy()
    e30 = frame["val30"].to_numpy()
    return dict(
        label=label,
        mean=float(pr.mean()),
        std=float(pr.std(ddof=1)),
        best=float(frame["best_epoch"].mean()),
        val30=float(e30.mean()),
        val30_std=float(e30.std(ddof=1)),
        drop=float(frame["drop"].mean()),
    )


def table(results):
    heads = ["후보", "PR-AUC", "최고에폭 평균", "30에폭 검증PR", "검증 하락"]
    aligns = ["left", "right", "right", "right", "right"]
    body = [[
        r["label"],
        f"{r['mean']:.4f}±{r['std']:.4f}",
        f"{r['best']:.1f}",
        f"{r['val30']:.4f}±{r['val30_std']:.4f}",
        f"{r['drop']:+.4f}",
    ] for r in results]
    widths = [max(fc.disp_width(row[i]) for row in [heads] + body) for i in range(len(heads))]
    lines = [heads] + body
    return "\n".join(
        "  ".join(fc.pad(row[i], widths[i], aligns[i]) for i in range(len(row))) for row in lines
    )


def judgment(results):
    """1위와 드롭아웃 0.2의 차이가 큰 표준편차 이내면 동점 → 검증 하락이 가장 작은 후보."""
    best = max(results, key=lambda r: r["mean"])
    drop = next(r for r in results if r["label"] == "드롭아웃 0.2")
    diff = best["mean"] - drop["mean"]
    bound = max(best["std"], drop["std"])
    if diff <= bound:
        chosen = min(results, key=lambda r: (r["drop"], -r["mean"]))
        tail = f"동점, 검증 하락이 가장 작은 {chosen['label']} 선택"
    else:
        tail = f"선택 {best['label']}"
    return (f"판정: PR-AUC 1위 {best['label']}, 드롭아웃 0.2와의 차이 {diff:+.4f}, "
            f"두 후보 표준편차 중 큰 값 {bound:.4f} → {tail}")


def main():
    _, start, _ = prev()
    specs = [
        ("없음(출발)", start),
        ("드롭아웃 0.2", fc.cfg_of(**dict(start, dropout=0.2))),
        ("L2 1e-4", fc.cfg_of(**dict(start, l2=1e-4))),
        ("단순화 (14,28,14)", fc.cfg_of(**dict(start, hidden=(14, 28, 14)))),
    ]
    results = [summarize(label, rows_of(cfg)) for label, cfg in specs]
    print("=== 과적합 대응 기법 비교 (출발 조건에 기법 하나만 추가) ===")
    print(fc.cfg_line(start, skip=("dropout", "l2", "early_stop")))
    print(table(results))
    print(judgment(results))


if __name__ == "__main__":
    main()
