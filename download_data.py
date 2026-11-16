"""Download Binance spot klines (monthly bulk dumps) into one Parquet file per symbol.

Run once by hand. Never call this from the agent loop.

    python download_data.py                              # BTCUSDT + ETHUSDT, 1h, 2019-01 .. 2024-12
    python download_data.py --holdout --end 2026-10      # Week 5 only, after strategies are frozen
"""
import argparse
import hashlib
import io
import json
import subprocess
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
KEEP = ["ts", "open", "high", "low", "close", "volume", "quote_volume", "trades"]
# Gemma 4's training data ends January 2025. Nothing from this month onward may be
# on disk until the holdout is opened (PLAN.md, "Splits"). Change only with the model.
LOCKED_FROM = "2025-01"
DATA = Path("data")


def check_unlocked(months, holdout, cwd="."):
    if not holdout and any(m >= LOCKED_FROM for m in months):
        raise SystemExit(f"Months from {LOCKED_FROM} on are the locked holdout. "
                         "Pass --holdout only when opening it (PLAN.md, Week 5).")
    if holdout and not subprocess.run(["git", "tag", "--list", "prereg-v1"], cwd=cwd,
                                      capture_output=True, text=True).stdout.strip():
        raise SystemExit("The holdout cannot be downloaded before the test is pre-registered: "
                         "commit PREREGISTRATION.md and run `git tag prereg-v1` first.")


def parse_klines(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, header=None, names=COLS)
    # A header row, if a file has one, turns into NaN here and is dropped.
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["open_time"])
    t = df["open_time"].astype("int64")
    t = t.where(t < 10**14, t // 1000)  # spot files from 2025-01 are microseconds, earlier ones milliseconds
    df["ts"] = pd.to_datetime(t, unit="ms", utc=True)
    return df[KEEP].reset_index(drop=True)


def fetch_month(symbol, interval, month, manifest):
    """Return the month's zip bytes (cached under data/raw), or None if Binance has no such file."""
    name = f"{symbol}-{interval}-{month}.zip"
    raw = DATA / "raw" / name
    if raw.exists():
        return raw.read_bytes()
    url = f"{BASE}/{symbol}/{interval}/{name}"
    try:
        body = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    expected = urllib.request.urlopen(url + ".CHECKSUM", timeout=60).read().split()[0].decode()
    digest = hashlib.sha256(body).hexdigest()
    if digest != expected:
        raise ValueError(f"checksum mismatch for {name}")
    raw.write_bytes(body)
    manifest[name] = {"sha256": digest,
                      "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return body


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--interval", default="1h")
    p.add_argument("--start", default="2019-01")
    p.add_argument("--end", default="2024-12")
    p.add_argument("--holdout", action="store_true",
                   help="allow locked months and write to data/holdout instead of data/klines")
    args = p.parse_args(argv)

    months = [str(m) for m in pd.period_range(args.start, args.end, freq="M")]
    check_unlocked(months, args.holdout)
    out_dir = DATA / ("holdout" if args.holdout else "klines")
    for d in (DATA / "raw", out_dir):
        d.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for symbol in args.symbols:
        frames, missing = [], []
        for month in months:
            body = fetch_month(symbol, args.interval, month, manifest)
            if body is None:
                missing.append(month)
            else:
                frames.append(parse_klines(body))
        manifest_path.write_text(json.dumps(manifest, indent=1))
        if not frames:
            print(f"{symbol}: no data in range")
            continue
        df = pd.concat(frames).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        gaps = int((df["ts"].diff().dropna() != pd.Timedelta(args.interval)).sum())
        df.to_parquet(out_dir / f"{symbol}_{args.interval}.parquet", index=False)
        print(f"{symbol}: {len(df)} bars, {df.ts.iloc[0]} .. {df.ts.iloc[-1]}, "
              f"{gaps} gaps, {len(missing)} months not on server {missing[:3]}")


if __name__ == "__main__":
    main()
