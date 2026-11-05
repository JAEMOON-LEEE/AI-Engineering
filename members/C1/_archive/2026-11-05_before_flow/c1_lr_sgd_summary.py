"""
lr_sgd_curve.csv를 읽어 학습률 그리드 요약을 터미널에 출력한다.
다시 학습하지 않는다.

사용법:
  python c1_lr_sgd_summary.py
"""

from __future__ import annotations

import os
import unicodedata

import numpy as np
import pandas as pd

C1_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(C1_DIR, "figures", "lr_sgd_curve.csv")
LRS = [0.00005, 0.001, 0.01, 0.1]
LABELS = ["lr=0.00005", "lr=0.001", "lr=0.01", "lr=0.1"]
HEADERS = ["후보", "PR-AUC", "학습PR", "차이", "재현율", "양성예측", "최고/전체에폭"]
ALIGNS = ["left", "right", "right", "right", "right", "right", "right"]


def disp_width(text):
    width = 0
    for ch in text:
        width += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
    return width


def pad(text, width, align):
    gap = width - disp_width(text)
    if gap <= 0:
        return text
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def row_at_best(sub):
    i = sub["val_pr"].idxmax()
    return sub.loc[i]


def summarize(df, lr):
    sub = df[np.isclose(df["lr"], lr)]
    best = [row_at_best(one) for _, one in sub.groupby("seed")]
    val = np.array([float(r.val_pr) for r in best])
    train = np.array([float(r.train_pr) for r in best])
    recall = np.array([float(r.val_recall) for r in best])
    pos = np.array([float(r.val_pos_rate) for r in best])
    epoch = np.array([float(r.epoch) for r in best])
    val_mean = float(val.mean())
    train_mean = float(train.mean())
    return [
        LABELS[LRS.index(lr)],
        f"{val_mean:.4f}±{float(val.std(ddof=1)):.4f}",
        f"{train_mean:.3f}",
        f"{train_mean - val_mean:+.3f}",
        f"{float(recall.mean()):.3f}",
        f"{float(pos.mean()):.3f}",
        f"{int(round(float(epoch.mean())))}/30",
    ]


def render(table):
    aligns = ["left"] + ["right"] * (len(HEADERS) - 1)
    widths = [
        max(disp_width(HEADERS[i]), *(disp_width(row[i]) for row in table))
        for i in range(len(HEADERS))
    ]
    lines = []
    for cells in (HEADERS, *table):
        lines.append("  ".join(pad(cells[i], widths[i], ALIGNS[i]) for i in range(len(cells))))
    return "\n".join(lines)


def main():
    df = pd.read_csv(CSV_PATH)
    table = [summarize(df, lr) for lr in LRS]
    print("=== 학습률 그리드 (SGD+NAG, 그 외 0단계 그대로) ===")
    print("(28,56,28), SGD+NAG, 배치 128, 최대 30에폭, 조기종료 없음, 불균형 처리 없음")
    print(render(table))


if __name__ == "__main__":
    main()
