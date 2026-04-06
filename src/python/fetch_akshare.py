"""
Centralized akshare API calls with CSV caching.
Each function checks for a cached file before hitting the API.
"""

import time
import pandas as pd
import akshare as ak
from config import RAW_DIR

CACHE_DIR = RAW_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cached(name, fetch_fn, max_age_days=7):
    """Return cached DataFrame if fresh enough, otherwise fetch and cache."""
    path = CACHE_DIR / f"{name}.csv"
    if path.exists():
        import os, datetime
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        if (datetime.datetime.now() - mtime).days < max_age_days:
            return pd.read_csv(path)
    df = fetch_fn()
    df.to_csv(path, index=False)
    return df


def fetch_urban_fai():
    """Urban fixed asset investment from akshare."""
    def _fetch():
        df = ak.macro_china_gdzctz()
        df["月份"] = pd.to_datetime(df["月份"], format="%Y年%m月份", errors="coerce")
        return df
    return _cached("urban_fai", _fetch)


def fetch_fr007_daily(start_year=2000, end_year=2025):
    """Daily FR007 repo rates, fetched year by year."""
    def _fetch():
        all_data = []
        for year in range(start_year, end_year + 1):
            try:
                df = ak.repo_rate_hist(start_date=f"{year}0101", end_date=f"{year}1231")
                all_data.append(df)
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching FR007 for {year}: {e}")
        if not all_data:
            raise RuntimeError("No FR007 data fetched")
        full = pd.concat(all_data, ignore_index=True)
        if "日期" in full.columns:
            full.rename(columns={"日期": "date"}, inplace=True)
        full["date"] = pd.to_datetime(full["date"])
        full.sort_values("date", inplace=True)
        return full
    return _cached("fr007_daily", _fetch)


def fetch_cpr_daily():
    """Central parity rate (USD/CNY) from Bank of China via akshare."""
    def _fetch():
        df = ak.currency_boc_sina(symbol="美元", start_date="20000101", end_date="20251231")
        df = df[["日期", "央行中间价"]].rename(columns={"日期": "date", "央行中间价": "CNYUSDCPR"})
        df["date"] = pd.to_datetime(df["date"])
        return df
    return _cached("cpr_daily", _fetch)


def fetch_m2():
    """M2 money supply from akshare."""
    def _fetch():
        df = ak.macro_china_supply_of_money()
        df = df[["统计时间", "货币和准货币（广义货币M2）"]].rename(
            columns={"统计时间": "date", "货币和准货币（广义货币M2）": "M2"}
        )
        df["date"] = pd.to_datetime(df["date"])
        return df
    return _cached("m2_supply", _fetch)
