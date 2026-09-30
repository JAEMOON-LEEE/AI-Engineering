"""A의 공식 분할 CSV를 data/final/ 이름으로만 복사한다. 내용은 바꾸지 않는다."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "members" / "A" / "Shopping_Project"
OUT_DIR = PROJECT_ROOT / "data" / "final"

SOURCES = {
    "train": SRC_DIR / "train.csv",
    "validation": SRC_DIR / "validation.csv",
    "test": SRC_DIR / "test.csv",
}
OUTPUTS = {
    "train": OUT_DIR / "train_clean.csv",
    "validation": OUT_DIR / "validation_final.csv",
    "test": OUT_DIR / "test_final.csv",
}
EXPECTED_ROWS = {"train": 8543, "validation": 1221, "test": 2441}


def row_hashes(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df, index=False)


def purchase_rate(df: pd.DataFrame) -> float:
    return float(df["Revenue"].astype(int).mean())


def main() -> int:
    errors = []
    frames = {}

    for name, path in SOURCES.items():
        if not path.exists():
            errors.append(f"[{name}] 입력 파일이 없음: {path}")
            continue
        frames[name] = pd.read_csv(path)

    if len(frames) == 3:
        for name, expected in EXPECTED_ROWS.items():
            actual = len(frames[name])
            print(f"[행 수] {name}: {actual} (기대 {expected})")
            if actual != expected:
                errors.append(f"[행 수] {name}이 {actual}행이라 기대값 {expected}과 다름")

        hashes = {name: set(row_hashes(df)) for name, df in frames.items()}
        pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
        for a, b in pairs:
            n_overlap = len(hashes[a] & hashes[b])
            print(f"[겹침] {a} vs {b}: {n_overlap}개")
            if n_overlap != 0:
                errors.append(f"[겹침] {a}와 {b} 사이 행 전체 해시가 {n_overlap}개 겹침 (기대 0)")

        for name, df in frames.items():
            rate = purchase_rate(df)
            print(f"[구매 비율] {name}: {rate:.6f} ({rate * 100:.2f}%)")

    if errors:
        print("\n검증 실패. 출력 파일을 만들지 않음.")
        for item in errors:
            print(f"- {item}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in SOURCES.items():
        dst = OUTPUTS[name]
        dst.write_bytes(src.read_bytes())
        print(f"[복사] {src} -> {dst}")

    print("\n검증 통과. 공식 분할 파일을 복사했다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
