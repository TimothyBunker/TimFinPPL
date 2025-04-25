#!/usr/bin/env python
"""
Fetch spot‑ and perpetual‑swap OHLCV plus funding‑rate data for crypto assets (via CCXT),
merge them into a single Parquet file.  
Robust to Deribit’s gigantic options catalogue and returns a tz‑aware Date column so Pandas
slicing works out of the box.
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dateutil import parser as dateparser

try:
    import ccxt  # noqa: F401
except ImportError:
    sys.exit("ccxt is required – install it with `pip install ccxt`. ")

# ╭──────────────────────────────────────────────────────────────────────────╮
# ┃ CLI                                                                     ┃
# ╰──────────────────────────────────────────────────────────────────────────╯

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fetch crypto spot/perp/funding data via CCXT")
    p.add_argument("--tickers", required=True,
                   help="Comma‑separated list.  Examples:\n"
                        "  BTC/USD                      (same leg for spot & perp)\n"
                        "  BTC/USD:BTC-PERPETUAL        (explicit perp leg)\n"
                        "  BTC/USD,ETH/USD:ETH-PERP     (multi‑pair)")
    p.add_argument("--start_date", required=True)
    p.add_argument("--end_date",   required=True)
    p.add_argument("--timeframe",  default="1h")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--spot_exchange", default="kraken")
    p.add_argument("--perp_exchange", default="deribit")
    p.add_argument("--output", default="data/crypto/crypto_data.parquet")
    return p.parse_args()


def iso_ms(ts: str) -> int:  # UTC milliseconds
    return int(dateparser.isoparse(ts).timestamp() * 1000)


# ╭──────────────────────────────────────────────────────────────────────────╮
# ┃ Exchange helpers                                                        ┃
# ╰──────────────────────────────────────────────────────────────────────────╯

def boot(id_: str, *, kind: str):
    opts: Dict[str, Any] = {"enableRateLimit": True}
    if kind == "perp":
        opts["options"] = {"defaultType": "swap"}
    ex = getattr(ccxt, id_)(opts)
    ex.load_markets()
    print(f"✔ {id_} {kind} – {len(ex.symbols)} symbols")
    return ex


def build_resolver(ex: "ccxt.Exchange", *, kind: str):
    """Return symbol resolver that never grabs options legs."""
    # Deribit perpetuals end with "‑PERPETUAL";
    perpetuals = {s.split("-")[0]: s for s in ex.symbols if s.endswith("-PERPETUAL")}

    def _resolve(user: str) -> str:
        # 1) exact match
        if user in ex.symbols:
            return user
        base = user.split("/")[0].split("-")[0]
        if kind == "perp" and base in perpetuals:
            return perpetuals[base]
        # 2) first swap instrument that is not option/future
        for s in ex.symbols:
            m = ex.markets[s]
            if m.get("swap") and m.get("base") == base and not m.get("option"):
                return s
        raise ValueError(f"Cannot resolve symbol '{user}' on {ex.id}")

    return _resolve


# ╭──────────────────────────────────────────────────────────────────────────╮
# ┃ Data‑pull primitives                                                    ┃
# ╰──────────────────────────────────────────────────────────────────────────╯

def fetch_all(ex: "ccxt.Exchange", sym: str, since: int, tf: str, limit: int):
    out: List[List[Any]] = []
    cursor = since
    while True:
        batch = ex.fetch_ohlcv(sym, tf, since=cursor, limit=limit)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < limit:
            break
        cursor = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000)
    return out


def to_df(raw, col):
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    return df[["close"]].rename(columns={"close": col})


# ╭──────────────────────────────────────────────────────────────────────────╮
# ┃ Main                                                                    ┃
# ╰──────────────────────────────────────────────────────────────────────────╯

def main():
    a = cli()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = iso_ms(a.start_date), iso_ms(a.end_date)
    start_dt, end_dt = pd.to_datetime(a.start_date, utc=True), pd.to_datetime(a.end_date, utc=True)

    spot = boot(a.spot_exchange, kind="spot")
    perp = boot(a.perp_exchange, kind="perp")
    res_spot = build_resolver(spot, kind="spot")
    res_perp = build_resolver(perp, kind="perp")

    frames: List[pd.DataFrame] = []
    for pair in a.tickers.split(','):
        pair = pair.strip()
        leg_spot, leg_perp = (pair.split(':', 1) + [""])[:2]
        if not leg_perp:
            leg_perp = leg_spot

        s_sym = res_spot(leg_spot)
        p_sym = res_perp(leg_perp)
        label = leg_spot

        print(f"⏳ {label}: spot {s_sym}")
        spot_raw = fetch_all(spot, s_sym, start_ms, a.timeframe, a.limit)
        df_s = to_df(spot_raw, "spot_price")

        print(f"⏳ {label}: perp {p_sym}")
        try:
            perp_raw = fetch_all(perp, p_sym, start_ms, a.timeframe, a.limit)
        except Exception as exc:
            print(f"⚠ {p_sym} failed ({exc}) – substituting spot data")
            perp_raw = []
        df_p = to_df(perp_raw or spot_raw, "perp_price")

        # funding
        if hasattr(perp, "fetch_funding_rate_history"):
            try:
                fr = perp.fetch_funding_rate_history(p_sym, since=start_ms, limit=a.limit)
                df_f = pd.DataFrame(fr)[["timestamp", "fundingRate"]]
                df_f.index = pd.to_datetime(df_f.pop("timestamp"), unit="ms", utc=True)
                df_f = df_f.rename(columns={"fundingRate": "funding_rate"})
            except Exception:
                df_f = pd.DataFrame(index=df_p.index, columns=["funding_rate"]).fillna(0)
        else:
            df_f = pd.DataFrame(index=df_p.index, columns=["funding_rate"]).fillna(0)

        df = df_s.join(df_p, how="inner").join(df_f, how="left")
        df["basis"] = df["perp_price"] - df["spot_price"]
        df["Ticker"] = label
        frames.append(df.loc[start_dt:end_dt])

    final = pd.concat(frames).reset_index(names="Date")
    final.to_parquet(a.output)
    print(f"✅ {len(final):,} rows → {a.output}")


if __name__ == "__main__":
    main()
