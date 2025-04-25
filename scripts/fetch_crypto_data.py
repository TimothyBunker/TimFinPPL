#!/usr/bin/env python
"""
Fetch spot and perpetual futures OHLCV and funding-rate data for crypto assets via CCXT,
merge into a single DataFrame, and save as Parquet for downstream training.
"""
import argparse
import os
import time
try:
    import ccxt
    # Debug: show environment and ccxt
    import sys
    print(f"[Debug] Python executable: {sys.executable}")
    try:
        print(f"[Debug] ccxt version: {ccxt.__version__}")
    except Exception:
        print("[Debug] ccxt imported, version unknown")
except ImportError:
    ccxt = None
import pandas as pd
try:
    from dateutil import parser as dateparser
except ImportError:
    raise ImportError("python-dateutil is required. Install with `pip install python-dateutil`")

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch crypto spot/perp/funding data')
    parser.add_argument('--tickers', type=str, required=True,
                        help='Comma-separated list of symbols, e.g. BTC/USDT,ETH/USDT')
    parser.add_argument('--start_date', type=str, required=True,
                        help='ISO start date, e.g. 2022-01-01T00:00:00Z')
    parser.add_argument('--end_date', type=str, required=True,
                        help='ISO end date, e.g. 2022-06-01T00:00:00Z')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='CCXT OHLCV timeframe (e.g. 1h, 4h)')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Max OHLCV candles per fetch')
    parser.add_argument('--spot-exchange', type=str, default='binanceus',
                        help='CCXT spot exchange id (e.g. binanceus, coinbasepro)')
    parser.add_argument('--perp-exchange', type=str, default='deribit',
                        help='CCXT perpetual futures exchange id (e.g. deribit, binanceus)')
    parser.add_argument('--output', type=str, default='data/crypto/crypto_data.parquet',
                        help='Output Parquet file path')
    return parser.parse_args()

def iso_to_ms(ts: str) -> int:
    dt = dateparser.isoparse(ts)
    return int(dt.timestamp() * 1000)

def fetch_ohlcv(exchange, symbol, since, timeframe, limit):
    all_bars = []
    t0 = since
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=t0, limit=limit)
        if not bars:
            break
        all_bars.extend(bars)
        if len(bars) < limit:
            break
        t0 = bars[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
    return all_bars

def main():
    # Parse CLI args (shows help/usage first)
    args = parse_args()
    if ccxt is None:
        import sys
        print("Error: ccxt library is required. Install into your venv with `pip install ccxt`." )
        print(f"Python executable: {sys.executable}")
        exit(1)
    # Prepare spot exchange with fallback list
    spot_exch = None
    # Fallback priorities for spot: user choice, kraken, binanceus, binance, bitstamp
    spot_ids = [args.spot_exchange, 'kraken', 'binanceus', 'binance', 'bitstamp']
    for ex in spot_ids:
        try:
            spot_exch = getattr(ccxt, ex)({'enableRateLimit': True})
            spot_exch.load_markets()
            print(f"Using spot exchange: {ex}")
            break
        except Exception as e:
            print(f"Spot exchange '{ex}' failed: {e}")
            spot_exch = None
    if spot_exch is None:
        print("Error: no working spot exchange found. Install ccxt or choose --spot-exchange.")
        exit(1)
    # Prepare perpetual futures exchange (for funding rates)
    perp_exch = None
    # Fallback priorities: user choice, deribit, binanceus, bitmex
    perp_ids = [args.perp_exchange, 'deribit', 'binanceus', 'bitmex']
    for ex in perp_ids:
        try:
            perp_exch = getattr(ccxt, ex)({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            perp_exch.load_markets()
            print(f"Using perpetual exchange: {ex}")
            break
        except Exception as e:
            print(f"Perp exchange '{ex}' failed: {e}")
            perp_exch = None
    use_funding = perp_exch is not None
    if not use_funding:
        print("Warning: no working perpetual exchange found; funding rates will be zero.")
    since = iso_to_ms(args.start_date)
    # Iterate tickers
    records = []
    # Loop over ticker pairs: 'spot_symbol:perp_symbol'
    def resolve_symbol(exchange, sym):
        """Resolve a user-provided symbol to an exchange-supported one."""
        available = exchange.symbols
        # exact match
        if sym in available:
            return sym
        # uppercase/lowercase variants
        up = sym.upper(); low = sym.lower()
        if up in available:
            return up
        if low in available:
            return low
        # prefix match (common for perp: base-*)
        base = sym.split('/')[0].split('-')[0]
        for s in available:
            if s.startswith(base):
                return s
        # fallback to original
        return sym

    # Loop over ticker pairs: 'spot_symbol:perp_symbol'
    for pair in args.tickers.split(','):
        pair = pair.strip()
        # Split into spot and perp user strings
        if ':' in pair:
            user_spot, user_perp = pair.split(':', 1)
        else:
            user_spot = user_perp = pair
        # Resolve into actual exchange symbols
        spot_symbol = resolve_symbol(spot_exch, user_spot)
        perp_symbol = resolve_symbol(perp_exch, user_perp) if use_funding else user_perp
        label = user_spot  # user-facing name for DataFrame
        # Fetch spot OHLCV
        print(f"Fetching spot OHLCV for {spot_symbol}...")
        spot_bars = fetch_ohlcv(spot_exch, spot_symbol, since, args.timeframe, args.limit)
        df_spot = pd.DataFrame(spot_bars, columns=['timestamp','open','high','low','close','volume'])
        df_spot['Date'] = pd.to_datetime(df_spot['timestamp'], unit='ms')
        df_spot.set_index('Date', inplace=True)
        df_spot = df_spot[['close']].rename(columns={'close': 'spot_price'})
        # Fetch perp OHLCV
        print(f"Fetching perp OHLCV for {perp_symbol}...")
        use_symbol = perp_symbol
        try:
            perp_bars = fetch_ohlcv(perp_exch, perp_symbol, since, args.timeframe, args.limit)
        except Exception as e:
            # Attempt Deribit-specific suffix fallback: '-PERP' -> '-PERPETUAL'
            if 'BadSymbol' in str(e) and use_symbol.endswith('-PERP'):
                alt = use_symbol + 'ETUAL'
                print(f"Info: retrying perp OHLCV with alternate symbol '{alt}'")
                try:
                    perp_bars = fetch_ohlcv(perp_exch, alt, since, args.timeframe, args.limit)
                    use_symbol = alt
                except Exception:
                    print(f"Warning: cannot fetch perp OHLCV for {perp_symbol} or {alt}")
                    perp_bars = None
            else:
                print(f"Warning: cannot fetch perp OHLCV for {perp_symbol}: {e}")
                perp_bars = None
        if perp_bars:
            df_perp = pd.DataFrame(perp_bars, columns=['timestamp','open','high','low','close','volume'])
            df_perp['Date'] = pd.to_datetime(df_perp['timestamp'], unit='ms')
            df_perp.set_index('Date', inplace=True)
            df_perp = df_perp[['close']].rename(columns={'close': 'perp_price'})
        else:
            # Fallback: use spot prices as perp price
            df_perp = df_spot.rename(columns={'spot_price': 'perp_price'})
        # Funding rate history (perp)
        # Funding rate history
        # Funding rate history (perp)
        if use_funding:
            try:
                fr = perp_exch.fetch_funding_rate_history(use_symbol, since=since, limit=args.limit)
                df_fr = pd.DataFrame(fr)
                df_fr['Date'] = pd.to_datetime(df_fr['timestamp'], unit='ms')
                df_fr.set_index('Date', inplace=True)
                df_fr = df_fr[['fundingRate']]
            except Exception as e:
                print(f"Warning: failed to fetch funding rates for {perp_symbol}: {e}")
                df_fr = pd.DataFrame(index=df_spot.index)
                df_fr['fundingRate'] = 0.0
        else:
            df_fr = pd.DataFrame(index=df_spot.index)
            df_fr['fundingRate'] = 0.0
        # Merge spot, perp, funding
        df = df_spot.join(df_perp, how='inner').join(df_fr, how='left')
        # Use spot symbol as Ticker label
        df['Ticker'] = label
        df['basis'] = df['perp_price'] - df['spot_price']
        records.append(df)
    # Concat all
    result = pd.concat(records).reset_index().rename(columns={'index':'Date'})
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_parquet(args.output)
    print(f"Saved crypto data to {args.output}")

if __name__ == '__main__':
    main()