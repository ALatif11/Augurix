"""Write the discovery split: data/splits/discovery/, bars before 2023-01-01.

That folder is the only data mounted into the sandbox while agents run.
Validation needs earlier bars as indicator warm-up, so run_validation.py reads
data/klines (which ends 2024-12-31) and scores only bars from DISCOVERY_END on.

    python make_splits.py
"""
from pathlib import Path

import pandas as pd

DISCOVERY_END = pd.Timestamp("2023-01-01", tz="UTC")   # exclusive; validation starts here
VALIDATION_END = pd.Timestamp("2025-01-01", tz="UTC")  # exclusive; embargo and holdout start here
HOLDOUT_START = pd.Timestamp("2025-03-01", tz="UTC")   # first scored holdout bar; Jan-Feb 2025 is the embargo
DATA = Path("data")


def discovery(df):
    if df.ts.max() >= VALIDATION_END:
        raise ValueError(f"data/klines holds bars at or after {VALIDATION_END.date()}: the holdout has leaked")
    return df[df.ts < DISCOVERY_END]


def main():
    out = DATA / "splits" / "discovery"
    out.mkdir(parents=True, exist_ok=True)
    for src in sorted((DATA / "klines").glob("*.parquet")):
        part = discovery(pd.read_parquet(src))
        part.to_parquet(out / src.name, index=False)
        print(f"{src.name}: {len(part)} discovery bars, {part.ts.iloc[0]} .. {part.ts.iloc[-1]}")


if __name__ == "__main__":
    main()
