"""
Long-term projection data preparation.
Cells 57–66 of the original Data_Monthly notebook.
Output: china_longterm_data.csv
"""

import pandas as pd
from functools import reduce
from config import RAW_DIR, DERIVED_DIR
from fetch_akshare import fetch_m2


def load_m2():
    """M2 money supply with YoY growth."""
    m2_df = fetch_m2()
    m2_df["date"] = pd.to_datetime(m2_df["date"])
    m2_df = m2_df[m2_df["date"].dt.year >= 2000]
    m2_df = m2_df.sort_values("date").reset_index(drop=True)
    m2_df["M2_growth"] = m2_df["M2"] - m2_df["M2"].shift(12)
    return m2_df


def load_us_ip():
    """US Industrial Production YoY."""
    usip_df = pd.read_csv(
        RAW_DIR / "INDPRO_PC1.csv", encoding="gbk", sep=",", header=0, index_col=0, engine="python"
    )
    usip_df = usip_df.reset_index().rename(columns={"observation_date": "date", "INDPRO_PC1": "US_IP_yoy"})
    usip_df["date"] = pd.to_datetime(usip_df["date"], errors="coerce")
    return usip_df


def main():
    print("=" * 60)
    print("  prep_longterm: Building long-term projection dataset")
    print("=" * 60)

    gdp_monthly_df = pd.read_csv(DERIVED_DIR / "gdp_monthly_df.csv", parse_dates=["date"])

    # Load FR007 monthly (from narrative prep or re-derive)
    full_repo_df = pd.read_csv(DERIVED_DIR / "romer_china_data.csv", parse_dates=["date"])

    ip_df = pd.read_csv(
        RAW_DIR / "IP.csv", encoding="gbk", sep=",", header=2, index_col=0, engine="python"
    )
    ip_df = ip_df.iloc[:2, :].copy().T.rename_axis("date").reset_index()
    ip_df["date"] = pd.to_datetime(ip_df["date"], format="%Y年%m月", errors="coerce")
    ip_df.columns = ["date", "IP_yoy", "IP_cumulative"]

    m2_df = load_m2()
    usip_df = load_us_ip()

    neer_df = pd.read_csv(
        RAW_DIR / "RBCNBIS_PC1.csv", encoding="gbk", sep=",", header=0, index_col=0, engine="python"
    )
    neer_df = neer_df.reset_index().rename(columns={"observation_date": "date", "RBCNBIS_PC1": "neer_yoy"})
    neer_df["date"] = pd.to_datetime(neer_df["date"], errors="coerce")

    ltp_df = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        [
            gdp_monthly_df[["date", "realgdp_monthly_yoy", "cpi", "CNYUSDSpot", "trade balance", "current consumption", "proxy_fai"]],
            full_repo_df[["date", "FR007"]],
            ip_df[["date", "IP_yoy"]],
            m2_df[["date", "M2_growth"]],
            usip_df[["date", "US_IP_yoy"]],
            neer_df[["date", "neer_yoy"]],
        ]
    ).sort_values("date").reset_index(drop=True)

    ltp_df.to_csv(DERIVED_DIR / "china_longterm_data.csv", index=False)
    print(f"Saved: {DERIVED_DIR / 'china_longterm_data.csv'}")
    return ltp_df


if __name__ == "__main__":
    main()
