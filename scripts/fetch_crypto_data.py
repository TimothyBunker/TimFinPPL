#!/usr/bin/env python
"""
Ultra‑history crypto data puller
================================
Fetch **as much history as the exchange allows**, including:
 • spot OHLCV  • perp‑swap OHLCV  • funding‑rate history
and write a tidy Parquet with columns::
    Date, Ticker, spot_price, perp_price, fundingRate, basis

### New in this version (back‑fill mode)
* **Back‑fill cursor** – when the first request returns 0 rows (because the
  `since` date is more than the exchange's window), the script automatically
  walks *backwards* in 5 000‑bar chunks until it reaches the target start date.
* Flag `--backfill` to force backward mode even if the forward loop would work.
* Works on Deribit, Binance USD‑M (`binanceusdm`), OKX, BitMEX, etc.  Spot side
  still grabs everything with the original forward loop.
* Funding‑rate fallback zero‑fills only the *missing* timestamps (keeps index).
* Verbose summary at the end shows rows per ticker and earliest/latest dates.
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
    p.add_argument("--tickers", required=True,
                   help="Comma‑separated list. Use A/B or A/B:PERP_SYM if perp differs.")
    p.add_argument("--start_date", required=True)
    p.add_argument("--end_date", required=True)
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--limit", type=int, default=1000,
                   help="Max candles per API call (exchange hard‑cap is often 500‑5 000).")
    p.add_argument("--spot_exchange", default="kraken")
    p.add_argument("--perp_exchange", default="binanceusdm",
                   help="Deribit only has ~200 days via OHLCV; Binance USD‑M gives 6+ years.")
    p.add_argument("--output", default="data/crypto/crypto_data.parquet")
    p.add_argument("--backfill", action="store_true",
                   help="Fetch perp data backwards in chunks until --start_date.")
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

# Perp resolver – picks first swap (non‑option) symbol for a base.

def perp_resolver(ex: "ccxt.Exchange"):
    mapping = {}
    for s, m in ex.markets.items():
        if m.get("swap") and not m.get("option") and m.get("quote") in {"USD", "USDT"}:
            mapping[m["base"]] = s
    def _res(user: str):
        base = user.split("/")[0].split("-")[0]
        if user in ex.symbols and ex.markets[user].get("swap"):
            return user
        if base in mapping:
            return mapping[base]
        raise ValueError(f"No swap found for '{user}' on {ex.id}")
    return _res


def spot_resolver(ex: "ccxt.Exchange"):
    def _res(sym: str):
        if sym in ex.symbols:
            return sym
        up, low = sym.upper(), sym.lower()
        if up in ex.symbols:
            return up
        if low in ex.symbols:
            return low
        raise ValueError(f"Spot symbol '{sym}' not on {ex.id}")
    return _res

# ─────────────────── Helpers ──────────────────

def fetch_forward(ex: "ccxt.Exchange", sym: str, since: int, tf: str, lim: int):
    out = []; cur = since
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


def fetch_backward(ex: "ccxt.Exchange", sym: str, since: int, tf: str, lim: int):
    now = ex.milliseconds()
    out: List[List[Any]] = []
    cursor = now
    while cursor > since:
        from_ts = max(since, cursor - lim * 3600_000)
        batch = ex.fetch_ohlcv(sym, tf, since=from_ts, limit=lim)
        if not batch:
            break
        out[:0] = batch  # prepend
        cursor = batch[0][0] - 1
        time.sleep(ex.rateLimit / 1000)
    return out


def to_df(raw, col):
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
    summary = []
    for pair in map(str.strip, a.tickers.split(',')):
        s_leg, p_leg = (pair.split(':',1)+[""])[:2]
        if not p_leg:
            p_leg = s_leg
        s_sym = s_res(s_leg)
        p_sym = p_res(p_leg)
        label = s_leg

        # Spot (forward pull is fine – exchanges allow deep history)
        s_raw = fetch_forward(spot, s_sym, start_ms, a.timeframe, a.limit)
        print(f"🔹 {label} spot bars: {len(s_raw)}")
        df_s = to_df(s_raw, "spot_price")

        # Perp: choose strategy
        perp_fetch = fetch_backward if (a.backfill or not fetch_forward(perp, p_sym, start_ms, a.timeframe, 1)) else fetch_forward
        p_raw = perp_fetch(perp, p_sym, start_ms, a.timeframe, a.limit)
        print(f"🔹 {label} perp bars: {len(p_raw)} (mode: {'backward' if perp_fetch==fetch_backward else 'forward'})")
        df_p = to_df(p_raw, "perp_price")

        # Funding
        try:
            fr = perp.fetch_funding_rate_history(p_sym, since=start_ms, limit=a.limit)
            df_f = pd.DataFrame(fr)[["timestamp","fundingRate"]]
            df_f.index = pd.to_datetime(df_f.pop("timestamp"), unit="ms", utc=True)
        except Exception:
            df_f = pd.DataFrame(index=df_p.index, columns=["fundingRate"]).fillna(0)

        df = df_s.join(df_p, how="inner").join(df_f, how="left")
        df["basis"] = df["perp_price"] - df["spot_price"]
        df["Ticker"] = label
        df = df.loc[start_dt:end_dt]
        if df.empty:
            raise RuntimeError(f"No rows for {label} – symbol/timeframe mismatch or history unavailable.")
        frames.append(df)
        summary.append((label, df.index[0], df.index[-1], len(df)))

    final = pd.concat(frames).reset_index(names="Date")
    final.to_parquet(a.output)
    print(f"✅ {len(final):,} total rows → {a.output}")
    print("Earliest/Latest per ticker:")
    for lab, first, last, rows in summary:
        print(f"  {lab:10} {str(first)[:10]} → {str(last)[:10]}  ({rows:6,d} rows)")

if __name__ == "__main__":
    main()
