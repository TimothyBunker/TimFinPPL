#!/usr/bin/env python
"""
Fetch spot OHLCV, perp‑swap OHLCV, and funding‑rate history and write a **non‑empty**
Parquet that downstream code (expects `fundingRate`) can consume.

Key fixes (2025‑04‑25):
• **Deribit resolver** now accepts the CCXT canonical symbol `BTC/USD:BTC` as well as
  `BTC-PERPETUAL`.  Both map to the same perpetual swap.  No more empty frames.
• **Row‑count guard** – aborts with a clear error if an asset yields zero rows after slicing.
• Writes funding‑rate column as `fundingRate` (camel‑case).
• Extra debug prints show how many bars were pulled per leg so you can see problems fast.
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dateutil import parser as dateparser

try:
    import ccxt
except ImportError:
    sys.exit("ccxt missing – pip install ccxt")

# ────────────────────────── CLI ──────────────────────────

def cli():
    p = argparse.ArgumentParser("Fetch crypto spot/perp/funding data via CCXT")
    p.add_argument("--tickers", required=True)
    p.add_argument("--start_date", required=True)
    p.add_argument("--end_date", required=True)
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--spot_exchange", default="kraken")
    p.add_argument("--perp_exchange", default="deribit")
    p.add_argument("--output", default="data/crypto/crypto_data.parquet")
    return p.parse_args()

iso_ms = lambda s: int(dateparser.isoparse(s).timestamp() * 1000)

# ─────────────────── Exchange bootstrap ──────────────────

def boot(eid: str, *, kind: str):
    opts: Dict[str, Any] = {"enableRateLimit": True}
    if kind == "perp":
        opts["options"] = {"defaultType": "swap"}
    ex = getattr(ccxt, eid)(opts)
    ex.load_markets()
    print(f"✔ {eid} {kind} – {len(ex.symbols)} symbols")
    return ex

# Deribit exposes two aliases for the swap: `BTC-PERPETUAL` *and* `BTC/USD:BTC`.

def perp_resolver(ex: "ccxt.Exchange"):
    mapping = {}
    for s in ex.symbols:
        if ex.markets[s].get("swap") and not ex.markets[s].get("option"):
            base = ex.markets[s]["base"]
            mapping[base] = s
    def _res(user: str):
        base = user.split("/")[0].split("-")[0]
        if base in mapping:
            return mapping[base]
        raise ValueError(f"No perpetual swap found for '{user}' on {ex.id}")
    return _res


def spot_resolver(ex: "ccxt.Exchange"):
    def _res(sym: str):
        if sym in ex.symbols:
            return sym
        for cand in (sym.upper(), sym.lower()):
            if cand in ex.symbols:
                return cand
        raise ValueError(f"Spot symbol '{sym}' not on {ex.id}")
    return _res

# ─────────────────── Helpers ──────────────────

def fetch_all(ex: "ccxt.Exchange", sym: str, since: int, tf: str, lim: int):
    out = []
    cur = since
    while True:
        batch = ex.fetch_ohlcv(sym, tf, since=cur, limit=lim)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < lim:
            break
        cur = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000)
    return out


def ohlcv_df(raw, col):
    df = pd.DataFrame(raw, columns=["ts","o","h","l","c","v"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    return df[["c"]].rename(columns={"c": col})

# ─────────────────── Main ──────────────────

def main():
    a = cli()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    start_ms = iso_ms(a.start_date)
    start_dt, end_dt = pd.to_datetime(a.start_date, utc=True), pd.to_datetime(a.end_date, utc=True)

    spot = boot(a.spot_exchange, kind="spot")
    perp = boot(a.perp_exchange, kind="perp")
    s_res, p_res = spot_resolver(spot), perp_resolver(perp)

    frames: List[pd.DataFrame] = []
    for pair in map(str.strip, a.tickers.split(',')):
        s_leg, p_leg = (pair.split(':',1)+[""])[:2]
        if not p_leg:
            p_leg = s_leg
        s_sym = s_res(s_leg)
        p_sym = p_res(p_leg)
        label = s_leg

        s_raw = fetch_all(spot, s_sym, start_ms, a.timeframe, a.limit)
        print(f"🔹 {label} spot bars: {len(s_raw)}")
        df_s = ohlcv_df(s_raw, "spot_price")
        try:
            p_raw = fetch_all(perp, p_sym, start_ms, a.timeframe, a.limit)
        except Exception as e:
            print(f"⚠ {p_sym} failed ({e}) – using spot as proxy")
            p_raw = []
        print(f"🔹 {label} perp bars: {len(p_raw)}")
        df_p = ohlcv_df(p_raw or s_raw, "perp_price")

        # funding
        try:
            fr = perp.fetch_funding_rate_history(p_sym, since=start_ms, limit=a.limit)
            df_f = pd.DataFrame(fr)[["timestamp","fundingRate"]]
            df_f.index = pd.to_datetime(df_f.pop("timestamp"), unit="ms", utc=True)
        except Exception as e:
            print(f"⚠ funding history for {p_sym}: {e}. Filling zeros")
            df_f = pd.DataFrame(index=df_p.index, columns=["fundingRate"]).fillna(0)

        df = df_s.join(df_p, how="inner").join(df_f, how="left")
        df["basis"] = df["perp_price"]-df["spot_price"]
        df["Ticker"] = label
        df_slice = df.loc[start_dt:end_dt]
        if df_slice.empty:
            raise RuntimeError(f"No rows for {label} after slicing – check symbols/timeframe.")
        frames.append(df_slice)

    final = pd.concat(frames).reset_index(names="Date")
    final.to_parquet(a.output)
    print(f"✅ Wrote {len(final):,} rows and {final.columns.tolist()} → {a.output}")

if __name__ == "__main__":
    main()
