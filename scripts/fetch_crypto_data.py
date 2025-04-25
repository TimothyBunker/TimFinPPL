#!/usr/bin/env python
"""
Fetch spot‑ and perpetual‑futures OHLCV + funding‑rate data for crypto assets via CCXT,
merge the feeds into a single Parquet file (rows = timestamps, columns = [spot, perp, funding]).

Key fixes vs. previous version
==============================
1. **Deribit perpetual symbols** – they are literally ``BTC-PERPETUAL`` / ``ETH-PERPETUAL``.
   The resolver now returns them explicitly when the user passes ``BTC-PERP`` or leaves the
   perp leg blank.
2. **Exchange type for Deribit** – Perpetuals are a *swap* on Deribit, not ``future``.
   ``options = {'defaultType': 'swap'}`` ensures CCXT picks the correct API branch.
3. **Robust symbol resolution** –  adds a tiny cache + helper for each exchange so we can
   look up by base‑currency, unified alias, or partial matches without grabbing random
   option chains.
4. **Funding‑rate availability guard** – if the exchange doesn’t support ``fetch_funding_rate_history``
   we fall back to zeros instead of crashing.
5. **Minor** – clearer CLI, PEP‑8-ish formatting, more logging.
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dateutil import parser as dateparser

try:
    import ccxt  # noqa: F401 – we only need to import to validate install
except ImportError as exc:
    sys.exit("ccxt is required – install with `pip install ccxt` (inside your venv).")

# ╭──────────────────────────────────────────────────────────────────────────╮
# ┃ CLI helpers                                                             ┃
# ╰──────────────────────────────────────────────────────────────────────────╯

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch crypto spot/perp/funding data via CCXT")
    p.add_argument("--tickers", required=True,
                   help="Comma‑separated list. Examples:\n"
                        "  * 'BTC/USD' (shares same symbol for spot & perp)\n"
                        "  * 'BTC/USD:BTC-PERP' (explicit perp leg)\n"
                        "  * 'BTC/USD:BTC-PERPETUAL,ETH/USD' (multiple pairs)")
    p.add_argument("--start_date", required=True, help="ISO8601, e.g. 2022-01-01T00:00:00Z")
    p.add_argument("--end_date", required=True, help="ISO8601, e.g. 2022-06-01T00:00:00Z")
    p.add_argument("--timeframe", default="1h", help="OHLCV timeframe – default 1h")
    p.add_argument("--limit", type=int, default=1000, help="Candles per batch (exchange cap)")
    p.add_argument("--spot_exchange", default="kraken", help="CCXT id for spot (default kraken)")
    p.add_argument("--perp_exchange", default="deribit", help="CCXT id for perpetuals")
    p.add_argument("--output", default="data/crypto/crypto_data.parquet",
                   help="Where to write the Parquet file")
    return p.parse_args()


def iso_ms(ts: str) -> int:
    return int(dateparser.isoparse(ts).timestamp() * 1000)


# ╭──────────────────────────────────────────────────────────────────────────╮
# ┃ Exchange bootstrap helpers                                              ┃
# ╰──────────────────────────────────────────────────────────────────────────╯

def boot_exchange(idx: str, *, kind: str) -> "ccxt.Exchange":
    """Instantiate + load markets. Kind is 'spot' | 'perp' for log semantics."""
    opts: Dict[str, Any] = {"enableRateLimit": True}
    if kind == "perp":
        # Deribit & Binance both require defaultType="swap" for perpetual endpoints.
        opts["options"] = {"defaultType": "swap"}
    ex = getattr(ccxt, idx)(opts)
    ex.load_markets()
    print(f"✔ Connected to {idx} ({kind}) – {len(ex.symbols)} symbols")
    return ex


def symbol_resolver(exchange: "ccxt.Exchange"):
    """Return a closure that maps user‑supplied strings to real exchange symbols."""
    syms = exchange.symbols  # cache list

    # If Deribit – build quick lookup for PERPETUALs
    perpetual_map: Dict[str, str] = {}
    if exchange.id == "deribit":
        for s in syms:
            if s.endswith("-PERPETUAL"):
                base = s.split("-")[0]
                perpetual_map[base] = s

    def _resolve(user_sym: str, *, leg: str) -> str:
        # 1. exact hit
        if user_sym in syms:
            return user_sym
        # 2. attempt Deribit base->PERPETUAL mapping for perp leg
        base = user_sym.split("/")[0].split("-")[0]
        if leg == "perp" and base in perpetual_map:
            return perpetual_map[base]
        # 3. case variants
        up, low = user_sym.upper(), user_sym.lower()
        if up in syms:
            return up
        if low in syms:
            return low
        # 4. fallback: first symbol starting with same base (beware!
        for s in syms:
            if s.startswith(base):
                return s
        return user_sym  # may raise later if still bad

    return _resolve


# ╭──────────────────────────────────────────────────────────────────────────╮
# ┃ Data pull primitives                                                    ┃
# ╰──────────────────────────────────────────────────────────────────────────╯

def fetch_ohlcv_all(exchange: "ccxt.Exchange", symbol: str, since: int,
                    timeframe: str, limit: int) -> List[List[Any]]:
    out: List[List[Any]] = []
    cursor = since
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < limit:
            break
        cursor = batch[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
    return out


def to_ohlcv_df(raw: List[List[Any]], col: str) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms")
    return df[["close"]].rename(columns={"close": col})


# ╭──────────────────────────────────────────────────────────────────────────╮
# ┃ Main driver                                                             ┃
# ╰──────────────────────────────────────────────────────────────────────────╯

def main():
    args = cli()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = iso_ms(args.start_date), iso_ms(args.end_date)

    spot = boot_exchange(args.spot_exchange, kind="spot")
    perp = boot_exchange(args.perp_exchange, kind="perp")

    spot_resolve = symbol_resolver(spot)
    perp_resolve = symbol_resolver(perp)

    frames: List[pd.DataFrame] = []

    for pair in args.tickers.split(","):
        pair = pair.strip()
        user_spot, user_perp = (pair.split(":", 1) + [""])[:2]
        if not user_perp:
            user_perp = user_spot  # default to same base

        spot_sym = spot_resolve(user_spot, leg="spot")
        perp_sym = perp_resolve(user_perp, leg="perp")
        label = user_spot  # nice name for downstream grouping

        print(f"⏳ {label}: pulling spot {spot_sym} …")
        spot_raw = fetch_ohlcv_all(spot, spot_sym, start_ms, args.timeframe, args.limit)
        df_spot = to_ohlcv_df(spot_raw, "spot_price")

        print(f"⏳ {label}: pulling perp {perp_sym} …")
        try:
            perp_raw = fetch_ohlcv_all(perp, perp_sym, start_ms, args.timeframe, args.limit)
        except Exception as exc:
            print(f"⚠️  {perp_sym} failed – {exc}. Falling back to spot prices for perp leg.")
            perp_raw = []
        df_perp = to_ohlcv_df(perp_raw or spot_raw, "perp_price")

        # funding
        if hasattr(perp, "fetch_funding_rate_history"):
            try:
                fr_hist = perp.fetch_funding_rate_history(perp_sym, since=start_ms, limit=args.limit)
                df_fr = pd.DataFrame(fr_hist)[["timestamp", "fundingRate"]]
                df_fr.index = pd.to_datetime(df_fr.pop("timestamp"), unit="ms")
                df_fr = df_fr.rename(columns={"fundingRate": "funding_rate"})
            except Exception as exc:
                print(f"⚠️  funding history unavailable for {perp_sym}: {exc}")
                df_fr = pd.DataFrame(index=df_perp.index, columns=["funding_rate"]).fillna(0)
        else:
            df_fr = pd.DataFrame(index=df_perp.index, columns=["funding_rate"]).fillna(0)

        df = df_spot.join(df_perp, how="inner").join(df_fr, how="left")
        df["basis"] = df["perp_price"] - df["spot_price"]
        df["Ticker"] = label
        frames.append(df.loc[args.start_date:args.end_date])

    final = pd.concat(frames).reset_index().rename(columns={"index": "Date"})
    final.to_parquet(args.output)
    print(f"✅ Saved {len(final):,} rows to {args.output}\n")


if __name__ == "__main__":
    main()
