"""
ETF Portfolio Performance Analyzer
====================================
Downloads historical price data (from a configurable start date) for a
list of funds/ETFs from Yahoo Finance, plots their cumulative growth
(normalized to 100 at the start), compares them against the S&P 500
(SPY) as a benchmark, and prints summary stats (total return, CAGR,
annualized volatility, max drawdown).

Requirements:
    pip install yfinance pandas matplotlib numpy

Usage:
    1. Edit the TICKERS dict below with your own {ticker: display_name} pairs.
    2. Run:  python etf_portfolio_analysis.py
    3. A chart window will open and a PNG will be saved to
       'etf_portfolio_growth.png' in the same folder.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) CONFIGURATION — edit this section for your own portfolio
# ----------------------------------------------------------------------

# Put your own tickers here as {ticker: display_name}.
# The display_name is what will show up in the legend and stats table.
TICKERS = {
    "0P0000WN7H.L": "HSBC American Index C Acc",
    "0P00000RGK.L": "Janus Henderson Glb Tech Leaders I Acc",
    "0P0000GBS1.L": "M&G Global Dividend Fund",
    "0P0000KANP.L": "UWS Blackrock Gold & General",
    "0P0001QIDE.L": "Invesco Emerging Markets ex China Fund (UK)",
    "0P0001PVQ3.L": "JPM Emerging Europe Equity II C Net Acc",
    "AIAFX": "abrdn Global Infrastructure A",
    "0P0000XOMV.L": "abrdn Latin American Equity I Acc",
    "0P00000L5P.L": "Schroder Sustainable UK Equity A Inc",
    "0P00007Y05.L": "Allianz Emerging Markets Equity Fund",
    "0P0000WN7N.L": "HSBC Pacific Index Accumulation C",
    "0P00013P6I.L": "HSBC FTSE All-World Index C Acc",
    "0P00000GAW.L": "Janus Henderson Global Fncls A Acc",
    "0P0000Q75E.L": "abrdn MyFolio Multi-Manager V Inst Acc",
    "0P0000WN1S.L": "M&G Asian GBP I Acc",
    "0P0000W46C.L": "JPM Multi-Asset Income C Net Acc",
    "0P0000W36K.L": "Artemis Global Income I Acc",
    "0P00016AYF.L": "abrdn Emerging Markets Bond X Acc GBP",
    "0P00013YAP.L": "Artemis US Smaller Companies I Acc GBP",
    "0P0000X9F9.L": "JPM Europe Dynamic (ex-UK) Fund C - Net Accumulation",
}

# Benchmark to compare against
BENCHMARK_TICKER = "SPY"
BENCHMARK_NAME = "S&P 500 (SPY)"

# Data start date: only pull history from this date onwards
START_DATE = "2026-01-01"

# ----------------------------------------------------------------------
# 2) DOWNLOAD DATA
# ----------------------------------------------------------------------

def download_prices(tickers, start_date):
    """
    Downloads adjusted close prices for a list of tickers, from
    start_date onwards. Returns a DataFrame with one column per ticker
    (dates as index). Each ticker is downloaded individually (via
    yf.Ticker().history()) so that a missing/failed ticker doesn't
    affect the others, and to avoid yfinance's multi-index column
    quirks that yf.download() can produce.
    """
    all_series = {}
    for ticker in tickers:
        print(f"Downloading {ticker} ...")
        try:
            data = yf.Ticker(ticker).history(start=start_date, auto_adjust=True)
        except Exception as e:
            print(f"  WARNING: failed to download {ticker} ({e}), skipping.")
            continue
        if data is None or data.empty or "Close" not in data.columns:
            print(f"  WARNING: no data returned for {ticker}, skipping.")
            continue
        close = data["Close"].copy()
        close.index = pd.to_datetime(close.index).tz_localize(None)  # drop tz for clean alignment
        close.name = ticker
        all_series[ticker] = close

    if not all_series:
        return pd.DataFrame()

    # pd.concat aligns all series on their date index (union), which is
    # safer than pd.DataFrame(dict) when series have different lengths/tz.
    prices = pd.concat(all_series.values(), axis=1)
    prices.columns = list(all_series.keys())
    prices = prices.sort_index()
    return prices


# Ticker -> display name lookup, used everywhere we print or plot labels
NAME_MAP = dict(TICKERS)
NAME_MAP[BENCHMARK_TICKER] = BENCHMARK_NAME

all_tickers = list(TICKERS.keys()) + [BENCHMARK_TICKER]
prices = download_prices(all_tickers, START_DATE)

if prices.empty:
    raise SystemExit("No data was downloaded. Check your tickers and internet connection.")

print("\nData downloaded. Date ranges available per ticker:")
for col in prices.columns:
    series = prices[col].dropna()
    if not series.empty:
        print(f"  {NAME_MAP.get(col, col)} ({col}): {series.index.min().date()} to {series.index.max().date()}")

# ----------------------------------------------------------------------
# 3) NORMALIZE TO GROWTH OF 100 (each series starts at its own inception)
# ----------------------------------------------------------------------

def normalize_to_100(df):
    """
    Normalizes each column so it starts at 100 on its own first
    available date. This lets you compare growth (%) even when
    ETFs have different inception dates.
    """
    normalized = df.copy()
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        first_valid = series.index[0]
        normalized[col] = df[col] / df.loc[first_valid, col] * 100
    return normalized


growth = normalize_to_100(prices)

# ----------------------------------------------------------------------
# 4) SUMMARY STATISTICS
# ----------------------------------------------------------------------

def summary_stats(prices, benchmark_col, name_map):
    """
    Computes total return, CAGR, annualized volatility, and max
    drawdown for each ticker in the price DataFrame.
    """
    rows = []
    for col in prices.columns:
        series = prices[col].dropna()
        if len(series) < 2:
            continue
        total_return = series.iloc[-1] / series.iloc[0] - 1
        n_years = (series.index[-1] - series.index[0]).days / 365.25
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
        daily_returns = series.pct_change().dropna()
        ann_vol = daily_returns.std() * np.sqrt(252)
        running_max = series.cummax()
        drawdown = series / running_max - 1
        max_dd = drawdown.min()
        rows.append({
            "Name": name_map.get(col, col),
            "Ticker": col,
            "Start": series.index[0].date(),
            "End": series.index[-1].date(),
            "Years": round(n_years, 1),
            "Total Return %": round(total_return * 100, 1),
            "CAGR %": round(cagr * 100, 2),
            "Ann. Volatility %": round(ann_vol * 100, 2),
            "Max Drawdown %": round(max_dd * 100, 1),
            "Benchmark": "Yes" if col == benchmark_col else "No",
        })
    return pd.DataFrame(rows).set_index("Name")


stats = summary_stats(prices, BENCHMARK_TICKER, NAME_MAP)
print("\nSummary statistics:\n")
print(stats.to_string())

# ----------------------------------------------------------------------
# 5) PLOT
# ----------------------------------------------------------------------

plt.figure(figsize=(14, 8))

for col in growth.columns:
    series = growth[col].dropna()
    label = NAME_MAP.get(col, col)
    if col == BENCHMARK_TICKER:
        plt.plot(series.index, series.values, label=f"{label} — benchmark",
                  color="black", linewidth=2.2, linestyle="--")
    else:
        plt.plot(series.index, series.values, label=label, linewidth=1.6)

plt.title(f"Portfolio Growth vs S&P 500 Benchmark\n(Value of 100 invested on {START_DATE})")
plt.xlabel("Date")
plt.ylabel("Growth of 100 (log scale)")
plt.yscale("log")  # log scale makes long-run compounding easier to compare
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, borderaxespad=0)
plt.grid(True, which="both", linestyle=":", alpha=0.5)
plt.tight_layout()

output_path = "etf_portfolio_growth.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nChart saved to: {output_path}")

plt.show()