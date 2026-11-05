"""
저장된 학습 곡선 CSV로 발표 슬라이드용 그림 2장을 그린다.
모델은 다시 학습하지 않는다.

  python c1_slide_figures.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

C1_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(C1_DIR, "figures")
CURVE30 = os.path.join(FIG_DIR, "initial_curve.csv")
CURVE300 = os.path.join(FIG_DIR, "initial_curve_e300.csv")
OUT30 = os.path.join(FIG_DIR, "slide_epoch30.png")
OUT300 = os.path.join(FIG_DIR, "slide_epoch300.png")
OPT_LR_CSV = os.path.join(FIG_DIR, "opt_lr_curve.csv")
OUT_HEATMAP = os.path.join(FIG_DIR, "opt_lr_heatmap.png")
OPT_LR_CW_CSV = os.path.join(FIG_DIR, "opt_lr_cw_curve.csv")
OUT_HEATMAP_CW = os.path.join(FIG_DIR, "opt_lr_cw_heatmap.png")
FONT_DIR = os.path.join(FIG_DIR, "fonts")

SEEDS = (42, 43, 44)
OFFICIAL30 = {42: 0.321157, 43: 0.167873, 44: 0.295385}

def seed_values(df, epoch, col):
    sub = df[df["epoch"] == epoch]
    return {int(seed): float(value) for seed, value in zip(sub["seed"], sub[col])}


def mean_at(df, epoch, col):
    return float(df.loc[df["epoch"] == epoch, col].mean())


def best_epochs(df):
    found = {}
    for seed, sub in df.groupby("seed"):
        found[int(seed)] = int(sub.loc[sub["val_pr"].idxmax(), "epoch"])
    return found


def check(df30, df300):
    """지정한 숫자가 아니면 False. 그래프는 그리지 않는다."""
    ok = True
    rows = []

    def add(name, got, expected, good):
        nonlocal ok
        ok = ok and good
        rows.append((name, got, expected, "일치" if good else "불일치"))

    v30 = seed_values(df30, 30, "val_pr")
    v30b = seed_values(df300, 30, "val_pr")
    for seed in SEEDS:
        got = round(v30[seed], 6)
        add(f"30에폭 시드 {seed}", f"{got:.6f}", f"{OFFICIAL30[seed]:.6f}", got == OFFICIAL30[seed])
        same = round(v30b[seed], 6) == OFFICIAL30[seed]
        add(f"300파일 30에폭 시드 {seed}", f"{v30b[seed]:.6f}", f"{OFFICIAL30[seed]:.6f}", same)

    add("30에폭 검증 평균", f"{mean_at(df30, 30, 'val_pr'):.4f}", "0.2615",
        round(mean_at(df30, 30, "val_pr"), 4) == 0.2615)
    add("100에폭 검증 평균", f"{mean_at(df300, 100, 'val_pr'):.4f}", "0.4243",
        round(mean_at(df300, 100, "val_pr"), 4) == 0.4243)
    add("300에폭 검증 평균", f"{mean_at(df300, 300, 'val_pr'):.4f}", "0.5414",
        round(mean_at(df300, 300, "val_pr"), 4) == 0.5414)
    add("300에폭 학습 평균", f"{mean_at(df300, 300, 'train_pr'):.4f}", "0.5859",
        round(mean_at(df300, 300, "train_pr"), 4) == 0.5859)

    for label, df, last in (("30에폭 파일", df30, 30), ("300에폭 파일", df300, 300)):
        peaks = best_epochs(df)
        for seed in SEEDS:
            add(f"{label} 시드 {seed} 최고 에폭", str(peaks[seed]), str(last), peaks[seed] == last)

    print(f"{'항목':<28} {'계산값':>12} {'기준':>12}  결과")
    for name, got, expected, flag in rows:
        print(f"{name:<28} {got:>12} {expected:>12}  {flag}")
    return ok


def _epoch_means(df):
    g = df.groupby("epoch")
    epochs = g.size().index.to_numpy()
    return epochs, g["train_pr"].mean().to_numpy(), g["val_pr"].mean().to_numpy()


def _textbook_plot(df, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    epochs, train_mean, val_mean = _epoch_means(df)
    plt.figure(figsize=(10.6, 5.6), dpi=200)
    plt.plot(epochs, train_mean)
    plt.plot(epochs, val_mean)
    plt.title(title)
    plt.ylabel("PR-AUC")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot30(df):
    _textbook_plot(df, "Model PR-AUC (30 epochs)", OUT30)


def plot300(df):
    _textbook_plot(df, "Model PR-AUC (300 epochs)", OUT300)


SELECTED = ("Adam", 0.01)
HEAT_OPTS = ("SGD", "RMSprop", "Adam")
HEAT_OPT_LABEL = {"SGD": "SGD+NAG", "RMSprop": "RMSprop", "Adam": "Adam"}
HEAT_LRS = (0.00005, 0.001, 0.01, 0.1)
HEAT_LR_TEXT = {0.00005: "0.00005", 0.001: "0.001", 0.01: "0.01", 0.1: "0.1"}
HEAT_EXPECTED = {
    ("SGD", 0.00005): "0.2615±0.0821",
    ("SGD", 0.001): "0.5908±0.0020",
    ("SGD", 0.01): "0.6713±0.0019",
    ("SGD", 0.1): "0.6768±0.0084",
    ("RMSprop", 0.00005): "0.5412±0.0162",
    ("RMSprop", 0.001): "0.6617±0.0123",
    ("RMSprop", 0.01): "0.6839±0.0041",
    ("RMSprop", 0.1): "0.6658±0.0035",
    ("Adam", 0.00005): "0.5322±0.0093",
    ("Adam", 0.001): "0.6673±0.0043",
    ("Adam", 0.01): "0.6808±0.0065",
    ("Adam", 0.1): "0.6655±0.0055",
}
HEAT_EXPECTED_CW = {
    ("SGD", 0.00005): "0.3432±0.0742",
    ("SGD", 0.001): "0.6006±0.0095",
    ("SGD", 0.01): "0.6757±0.0046",
    ("SGD", 0.1): "0.6738±0.0229",
    ("RMSprop", 0.00005): "0.5355±0.0347",
    ("RMSprop", 0.001): "0.6625±0.0073",
    ("RMSprop", 0.01): "0.6835±0.0062",
    ("RMSprop", 0.1): "0.6389±0.0136",
    ("Adam", 0.00005): "0.5339±0.0429",
    ("Adam", 0.001): "0.6669±0.0191",
    ("Adam", 0.01): "0.6915±0.0087",
    ("Adam", 0.1): "0.6495±0.0275",
}
_GOTHIC_FILES = {
    "bold": "GothicA1-Bold.ttf",
    "extrabold": "GothicA1-ExtraBold.ttf",
}
_GOTHIC_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/gothica1/{name}"


def _pick_column(columns, aliases, role):
    lowered = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    names = ", ".join(str(col) for col in columns)
    raise SystemExit(f"[중단] {role} 열을 찾지 못했습니다. CSV 열: {names}")


def _best_val_text(df, cols, optimizer, lr):
    """시드별 최고 검증 PR-AUC의 평균±표본표준편차. c1_opt_lr.py와 같은 방식."""
    import numpy as np

    opt_col, lr_col, seed_col, epoch_col, val_col = cols
    sub = df[(df[opt_col] == optimizer) & np.isclose(df[lr_col].astype(float), lr)]
    best = []
    for _, one in sub.groupby(seed_col):
        one = one.sort_values(epoch_col)
        best.append(float(one.loc[one[val_col].idxmax(), val_col]))
    if len(best) != 3:
        return None, None, f"시드 {len(best)}개"
    values = np.asarray(best, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return mean, std, f"{mean:.4f}±{std:.4f}"


def _valid_ttf(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 1000:
        return False
    with open(path, "rb") as handle:
        return handle.read(4) in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1")


def _installed_gothic(weight):
    from matplotlib import font_manager

    want = {"bold": 700, "extrabold": 800}[weight]
    for font in font_manager.fontManager.ttflist:
        if font.name.replace(" ", "").lower() != "gothica1":
            continue
        if int(font.weight) == want:
            return font.fname
    return None


def _download_gothic(filename, dest):
    import urllib.request

    url = _GOTHIC_URL.format(name=filename)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    with open(dest, "wb") as handle:
        handle.write(data)
    if not _valid_ttf(dest):
        os.remove(dest)
        raise SystemExit(f"[중단] 폰트 파일이 올바르지 않습니다: {url}")


def _gothic_paths():
    paths = {}
    os.makedirs(FONT_DIR, exist_ok=True)
    for weight, filename in _GOTHIC_FILES.items():
        installed = _installed_gothic(weight)
        dest = os.path.join(FONT_DIR, filename)
        if installed:
            paths[weight] = installed
        elif _valid_ttf(dest):
            paths[weight] = dest
        else:
            print(f"[폰트] Gothic A1 {weight} 없음. 받는 중: {filename}", flush=True)
            _download_gothic(filename, dest)
            paths[weight] = dest
    return paths


def _heat_table(stats):
    rows = []
    width_name = max(len("후보"), *(len(HEAT_OPT_LABEL[opt] + " " + HEAT_LR_TEXT[lr]) for opt, lr in stats))
    rows.append(f"{'후보':<{width_name}}  PR-AUC")
    for (opt, lr), item in stats.items():
        name = f"{HEAT_OPT_LABEL[opt]} {HEAT_LR_TEXT[lr]}"
        rows.append(f"{name:<{width_name}}  {item['text']}")
    return "\n".join(rows)


def _heat_colors():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "opt_lr", ["#FBFBFC", "#DADEF6", "#90ACF6"]
    )


def _cell_color(cmap, value, optimizer, lr):
    if optimizer == SELECTED[0] and abs(lr - SELECTED[1]) < 1e-12:
        return "#FEF4A1"
    tone = (value - 0.25) / (0.70 - 0.25)
    tone = min(1.0, max(0.0, tone))
    return cmap(tone)


def _draw_heatmap(stats, font_paths, out_path=OUT_HEATMAP):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    for path in font_paths.values():
        font_manager.fontManager.addfont(path)

    def face(weight, size):
        prop = font_manager.FontProperties(fname=font_paths[weight])
        prop.set_size(size)
        return prop

    fig = plt.figure(figsize=(10.6, 5.2), dpi=200)
    fig.patch.set_facecolor("#F5F6F7")
    ax = fig.add_axes([0.22, 0.16, 0.74, 0.70])
    ax.set_facecolor("#F5F6F7")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    pad_x, pad_y = 0.016, 0.022
    gap_x, gap_y = 0.020, 0.040
    cols, rows = len(HEAT_LRS), len(HEAT_OPTS)
    cell_w = (1 - 2 * pad_x - gap_x * (cols - 1)) / cols
    cell_h = (1 - 2 * pad_y - gap_y * (rows - 1)) / rows
    cmap = _heat_colors()
    number_face = face("extrabold", 36)
    row_face = face("extrabold", 18)
    col_face = face("bold", 16)
    axis_face = face("bold", 18)

    for row, optimizer in enumerate(HEAT_OPTS):
        y = pad_y + (rows - 1 - row) * (cell_h + gap_y)
        ax.text(
            -0.025, y + cell_h / 2, HEAT_OPT_LABEL[optimizer],
            transform=ax.transAxes, ha="right", va="center",
            fontproperties=row_face, color="#31343A", clip_on=False,
        )
        for col, lr in enumerate(HEAT_LRS):
            x = pad_x + col * (cell_w + gap_x)
            value = stats[(optimizer, lr)]["mean"]
            ax.add_patch(Rectangle(
                (x, y), cell_w, cell_h, transform=ax.transAxes,
                facecolor=_cell_color(cmap, value, optimizer, lr),
                edgecolor="#31343A", linewidth=3, joinstyle="miter", zorder=2,
            ))
            ax.text(
                x + cell_w / 2, y + cell_h / 2, f"{value:.3f}",
                transform=ax.transAxes, ha="center", va="center",
                fontproperties=number_face, color="#31343A", zorder=3,
            )
            if row == 0:
                ax.text(
                    x + cell_w / 2, 1.03, HEAT_LR_TEXT[lr],
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontproperties=col_face, color="#31343A", clip_on=False,
                )

    ax.text(
        0.5, -0.12, "학습률",
        transform=ax.transAxes, ha="center", va="top",
        fontproperties=axis_face, color="#4A4E55", clip_on=False,
    )
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_opt_lr_heatmap(csv_path=OPT_LR_CSV, out_path=OUT_HEATMAP, expected=None):
    """opt_lr_curve.csv로 옵티마이저 × 학습률 히트맵을 그린다. 다시 학습하지 않는다."""
    if expected is None:
        expected = HEAT_EXPECTED
    df = pd.read_csv(csv_path)
    columns = _pick_column(df.columns, ("optimizer", "optimiser"), "옵티마이저"), \
        _pick_column(df.columns, ("lr", "learning_rate"), "학습률"), \
        _pick_column(df.columns, ("seed",), "시드"), \
        _pick_column(df.columns, ("epoch",), "에폭"), \
        _pick_column(df.columns, ("val_pr", "val_pr_auc", "val_prauc"), "검증 PR-AUC")
    print("[열] 옵티마이저={0}, 학습률={1}, 시드={2}, 에폭={3}, 검증 PR-AUC={4}".format(*columns))

    stats = {}
    mismatches = []
    for optimizer in HEAT_OPTS:
        for lr in HEAT_LRS:
            mean, std, text = _best_val_text(df, columns, optimizer, lr)
            stats[(optimizer, lr)] = {"mean": mean, "std": std, "text": text}
            want = expected[(optimizer, lr)]
            if text != want:
                mismatches.append((optimizer, lr, text, want, mean, std))

    print(_heat_table(stats))
    if mismatches:
        print("[중단] 소수 넷째 자리가 다릅니다. 그림은 저장하지 않습니다.")
        for optimizer, lr, text, expected, mean, std in mismatches:
            exp_mean, exp_std = (float(part) for part in expected.split("±"))
            name = f"{HEAT_OPT_LABEL[optimizer]} {HEAT_LR_TEXT[lr]}"
            mean_gap = "계산 불가" if mean is None else f"{mean - exp_mean:+.6f}"
            std_gap = "계산 불가" if std is None else f"{std - exp_std:+.6f}"
            print(f"  {name}: 계산 {text} / 기준 {expected} / 평균 차이 {mean_gap} / 표준편차 차이 {std_gap}")
        return 1

    print("[확인] 12칸 모두 소수 넷째 자리까지 일치")
    _draw_heatmap(stats, _gothic_paths(), out_path)
    print(f"저장: {out_path}")
    return 0


def plot_opt_lr_cw_heatmap():
    """클래스 가중치 그리드 CSV로 같은 히트맵을 그린다. 기존 그림은 덮어쓰지 않는다."""
    return plot_opt_lr_heatmap(OPT_LR_CW_CSV, OUT_HEATMAP_CW, HEAT_EXPECTED_CW)


def main():
    df30 = pd.read_csv(CURVE30)
    df300 = pd.read_csv(CURVE300)
    if not check(df30, df300):
        print("[중단] 기준 값과 다릅니다. 그래프는 쓰지 않습니다.")
        return 1
    plot30(df30)
    plot300(df300)
    print(f"저장: {OUT30}")
    print(f"      {OUT300}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
