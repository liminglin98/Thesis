"""
Monthly GDP estimation using Stock & Watson (2010) distribution method.
Cells 0–40 of the original Data_Monthly notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from config import PROJECT_ROOT, RAW_DIR, DERIVED_DIR
from fetch_akshare import fetch_urban_fai
from stock_watson_distribute import prepare_and_run


# ---------------------------------------------------------------------------
# Individual data loaders
# ---------------------------------------------------------------------------

def load_consumption():
    df = pd.read_csv(
        RAW_DIR / "Social Consumption Retail Sales.csv",
        encoding="gbk", sep=",", header=2, index_col=0, engine="python",
    )
    df = df.iloc[:2, :].copy().T.rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"], format="%Y年%m月", errors="coerce")
    df.columns = ["date", "current consumption", "cumulative consumption"]

    year = df["date"].dt.year
    month = df["date"].dt.month
    feb_cum = df.loc[month.eq(2)].set_index(year[month.eq(2)])["cumulative consumption"]
    for m in [1, 2]:
        mask = month.eq(m) & df["current consumption"].isna()
        df.loc[mask, "current consumption"] = year[mask].map(feb_cum) / 2
    return df


def load_trade():
    df = pd.read_csv(
        RAW_DIR / "Export-Import.csv",
        encoding="gbk", sep=",", header=2, index_col=0, engine="python",
    )
    df = df.iloc[[12], :].copy().T.rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"], format="%Y年%m月", errors="coerce")
    df.columns = ["date", "trade balance"]

    spot_df = pd.read_excel(RAW_DIR / "CNYUSD_Spot_Rate.xlsx")
    spot_df = spot_df[["指标名称", "CFETS:即期汇率:美元兑人民币"]].rename(
        columns={"指标名称": "date", "CFETS:即期汇率:美元兑人民币": "CNYUSDSpot"}
    )
    spot_df = spot_df.iloc[4:-2]
    spot_df["date"] = pd.to_datetime(spot_df["date"])
    spot_df = spot_df[spot_df["date"].dt.year >= 2000]
    spot_df["CNYUSDSpot"] = pd.to_numeric(spot_df["CNYUSDSpot"], errors="coerce")
    spot_df = spot_df.groupby(spot_df["date"].dt.to_period("M"))["CNYUSDSpot"].mean().reset_index()
    spot_df["date"] = spot_df["date"].dt.to_timestamp()

    df = df.merge(spot_df[["date", "CNYUSDSpot"]], on="date", how="left")
    df["trade balance"] = (
        pd.to_numeric(df["trade balance"], errors="coerce")
        * df["CNYUSDSpot"] / 1_000_000
    )
    df = df.drop(columns=["CNYUSDSpot"])
    return df, spot_df


def load_fai():
    fai_df = pd.read_csv(
        RAW_DIR / "Fixed Asset Investment.csv",
        encoding="gbk", sep=",", header=2, index_col=0, engine="python",
    )
    fai_df = fai_df.iloc[[0], :].copy().T.rename_axis("date").reset_index()
    fai_df["date"] = pd.to_datetime(fai_df["date"], format="%Y年%m月", errors="coerce")
    fai_df.columns = ["date", "fai growth"]

    urbanfai_df = fetch_urban_fai()
    if "月份" not in urbanfai_df.columns:
        urbanfai_df["月份"] = pd.to_datetime(urbanfai_df["月份"], format="%Y年%m月份", errors="coerce")
    else:
        urbanfai_df["月份"] = pd.to_datetime(urbanfai_df["月份"], errors="coerce")

    years = sorted(
        y for y in urbanfai_df["月份"].dt.year.dropna().astype(int).unique() if y >= 2000
    )
    jan_dates = pd.to_datetime([f"{y}-01-01" for y in years])
    missing_jan = jan_dates[~jan_dates.isin(urbanfai_df["月份"].values)]
    if len(missing_jan) > 0:
        urbanfai_df = pd.concat([urbanfai_df, pd.DataFrame({"月份": missing_jan})], ignore_index=True)
        urbanfai_df = urbanfai_df.sort_values("月份").reset_index(drop=True)

    urbanfai_df = urbanfai_df[["月份", "当月", "自年初累计"]].rename(
        columns={"月份": "date", "当月": "current", "自年初累计": "cumulative"}
    )
    year = urbanfai_df["date"].dt.year
    month = urbanfai_df["date"].dt.month
    feb_cum = urbanfai_df.loc[month.eq(2)].set_index(year[month.eq(2)])["cumulative"]
    for m in [1, 2]:
        mask = month.eq(m) & urbanfai_df["current"].isna()
        urbanfai_df.loc[mask, "current"] = year[mask].map(feb_cum) / 2

    # Merge and backcast
    _fai = fai_df[["date", "fai growth"]].copy()
    _fai["date"] = pd.to_datetime(_fai["date"], errors="coerce")
    _urb = urbanfai_df[["date", "current", "cumulative"]].copy()
    _urb["date"] = pd.to_datetime(_urb["date"], errors="coerce")
    _urb["cumulative"] = pd.to_numeric(_urb["cumulative"], errors="coerce")
    _urb["current"] = pd.to_numeric(_urb["current"], errors="coerce")

    proxyfai_df = _fai.merge(_urb, on="date", how="outer").sort_values("date").set_index("date")
    proxyfai_df["fai growth"] = pd.to_numeric(proxyfai_df["fai growth"], errors="coerce")

    # Find anchor year
    valid_mask = (
        proxyfai_df["cumulative"].notna()
        & (proxyfai_df["cumulative"] > 0)
        & (proxyfai_df.index.year >= 2008)
    )
    anchor_year = int(proxyfai_df.index[valid_mask].year.min())

    # Backcast cumulative
    for yr in range(anchor_year - 1, 1999, -1):
        for mo in range(2, 13):
            date_curr = pd.Timestamp(year=yr, month=mo, day=1)
            date_next = pd.Timestamp(year=yr + 1, month=mo, day=1)
            if date_curr not in proxyfai_df.index or date_next not in proxyfai_df.index:
                continue
            cum_next = proxyfai_df.loc[date_next, "cumulative"]
            growth_next = proxyfai_df.loc[date_next, "fai growth"]
            if pd.notna(cum_next) and cum_next > 0 and pd.notna(growth_next):
                proxyfai_df.loc[date_curr, "cumulative"] = cum_next / (1 + growth_next / 100)

    # Derive current from cumulative
    for yr in proxyfai_df.index.year.unique():
        for mo in range(1, 13):
            date = pd.Timestamp(year=int(yr), month=mo, day=1)
            if date not in proxyfai_df.index:
                continue
            if mo == 2:
                feb_cum_val = proxyfai_df.loc[date, "cumulative"]
                if pd.notna(feb_cum_val):
                    proxyfai_df.loc[date, "current"] = feb_cum_val / 2
                    jan = pd.Timestamp(year=int(yr), month=1, day=1)
                    if jan in proxyfai_df.index:
                        proxyfai_df.loc[jan, "current"] = feb_cum_val / 2
            elif mo > 2:
                if pd.isna(proxyfai_df.loc[date, "current"]):
                    prev = pd.Timestamp(year=int(yr), month=mo - 1, day=1)
                    if prev in proxyfai_df.index:
                        cum_this = proxyfai_df.loc[date, "cumulative"]
                        cum_prev = proxyfai_df.loc[prev, "cumulative"]
                        if pd.notna(cum_this) and pd.notna(cum_prev):
                            proxyfai_df.loc[date, "current"] = cum_this - cum_prev

    proxyfai_df.columns = ["fai_growth", "proxy_fai", "proxy_cumulative_fai"]
    return proxyfai_df


def load_government():
    gov_df = pd.read_csv(
        RAW_DIR / "NationalGovSpend.csv",
        encoding="gbk", sep=",", header=2, index_col=0, engine="python",
    )
    gov_df = gov_df.iloc[[0], :].copy().T.rename_axis("date").reset_index()
    gov_df["date"] = pd.to_datetime(gov_df["date"], format="%Y年%m月", errors="coerce")
    gov_df.columns = ["date", "gov_spend_ytd"]
    gov_df = gov_df.sort_values("date").reset_index(drop=True)
    gov_df["gov_spend_ytd"] = pd.to_numeric(gov_df["gov_spend_ytd"], errors="coerce")

    year = gov_df["date"].dt.year
    month = gov_df["date"].dt.month
    feb_cum = gov_df.loc[month.eq(2)].set_index(year[month.eq(2)])["gov_spend_ytd"]

    gov_df["gov_spend_current"] = np.nan
    for m in [1, 2]:
        mask = month.eq(m)
        gov_df.loc[mask, "gov_spend_current"] = year[mask].map(feb_cum) / 2

    ytd_by_date = gov_df.set_index("date")["gov_spend_ytd"]
    mar_plus = month.gt(2)
    prev_dates = gov_df.loc[mar_plus, "date"] - pd.DateOffset(months=1)
    gov_df.loc[mar_plus, "gov_spend_current"] = (
        gov_df.loc[mar_plus, "gov_spend_ytd"].values - prev_dates.map(ytd_by_date).values
    )

    # Fill Dec 2000-2009 from annual totals
    annualgov_df = pd.read_csv(
        RAW_DIR / "AnnualGovSpend.csv",
        encoding="gbk", sep=",", header=2, index_col=0, engine="python",
    )
    annualgov_df = annualgov_df.iloc[[0], :].copy().T.rename_axis("date").reset_index()
    annualgov_df["date"] = pd.to_datetime(annualgov_df["date"], format="%Y年", errors="coerce")
    annualgov_df.columns = ["date", "gov_spend_annual"]
    annualgov_df["gov_spend_annual"] = pd.to_numeric(annualgov_df["gov_spend_annual"], errors="coerce")
    annual_by_year = annualgov_df.set_index(annualgov_df["date"].dt.year)["gov_spend_annual"]

    ytd_by_date = gov_df.set_index("date")["gov_spend_ytd"]
    dec_mask = (gov_df["date"].dt.month == 12) & (gov_df["date"].dt.year.between(2000, 2009))
    for i in gov_df.index[dec_mask]:
        yr = gov_df.loc[i, "date"].year
        annual = annual_by_year.get(yr, np.nan)
        nov_date = pd.Timestamp(year=yr, month=11, day=1)
        nov_ytd = ytd_by_date.get(nov_date, np.nan)
        if pd.notna(annual):
            gov_df.loc[i, "gov_spend_ytd"] = annual
            gov_df.loc[i, "gov_spend_current"] = annual - nov_ytd if pd.notna(nov_ytd) else np.nan

    return gov_df


def load_cpi():
    frames = []
    for fname in ["cpi2015.csv", "cpi2016-2020.csv", "cpi2021-2025.csv"]:
        df = pd.read_csv(
            RAW_DIR / fname,
            encoding="gbk", sep=",", header=2, index_col=0, engine="python",
        )
        df = df.iloc[[0], :].copy().T.rename_axis("date").reset_index()
        df["date"] = pd.to_datetime(df["date"], format="%Y年%m月", errors="coerce")
        df.columns = ["date", "cpi"]
        df["cpi"] = df["cpi"] - 100
        frames.append(df)

    cpi_df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return cpi_df


def load_real_gdp():
    gdp_df = pd.read_excel(RAW_DIR / "China_RealGDP.xlsx")
    gdp_df = gdp_df[["指标名称", "中国:GDP:不变价:当季值", "中国:GDP:不变价:当季同比"]].rename(
        columns={"指标名称": "date", "中国:GDP:不变价:当季值": "realgdp", "中国:GDP:不变价:当季同比": "realgdpgrowth"}
    )
    gdp_df = gdp_df.iloc[4:-2]
    gdp_df["date"] = pd.to_datetime(gdp_df["date"]).dt.to_period("Q")
    gdp_df = gdp_df[gdp_df["date"].dt.year >= 2000]
    gdp_df["realgdp"] = pd.to_numeric(gdp_df["realgdp"], errors="coerce")
    gdp_df["realgdpgrowth"] = pd.to_numeric(gdp_df["realgdpgrowth"], errors="coerce")

    # Backcast
    gdp_df = gdp_df.sort_values("date").set_index("date")
    for q in sorted(gdp_df.index, reverse=True):
        if pd.isna(gdp_df.loc[q, "realgdp"]):
            q_plus4 = q + 4
            if q_plus4 in gdp_df.index:
                gdp_next = gdp_df.loc[q_plus4, "realgdp"]
                growth_next = gdp_df.loc[q_plus4, "realgdpgrowth"]
                if pd.notna(gdp_next) and pd.notna(growth_next):
                    gdp_df.loc[q, "realgdp"] = gdp_next / (1 + growth_next / 100)
    gdp_df = gdp_df.reset_index()

    # Build chained GDP
    gdp_df = gdp_df.sort_values("date").reset_index(drop=True)
    anchor_idx = gdp_df["realgdp"].last_valid_index()
    chain_gdp = gdp_df["realgdp"].copy()
    for i in range(anchor_idx - 4, -1, -1):
        growth_forward = gdp_df.loc[i + 4, "realgdpgrowth"]
        level_forward = chain_gdp.iloc[i + 4]
        if pd.notna(growth_forward) and pd.notna(level_forward):
            chain_gdp.iloc[i] = level_forward / (1 + growth_forward / 100)
    gdp_df["realgdp_chained"] = chain_gdp
    return gdp_df


def load_neer():
    neer_df = pd.read_csv(
        RAW_DIR / "RBCNBIS_PC1.csv", encoding="gbk", sep=",", header=0, index_col=0, engine="python"
    )
    neer_df = neer_df.reset_index().rename(columns={"observation_date": "date", "RBCNBIS_PC1": "neer_yoy"})
    neer_df["date"] = pd.to_datetime(neer_df["date"], errors="coerce")
    return neer_df


# ---------------------------------------------------------------------------
# Combine and run Stock-Watson
# ---------------------------------------------------------------------------

def combine_monthly(cpi_df, gov_df, proxyfai_df, trade_df, consump_df, spot_df, neer_df, gdp_df):
    gdp_monthly_df = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        [
            cpi_df[["date", "cpi"]],
            gov_df[["date", "gov_spend_current"]],
            proxyfai_df[["proxy_fai"]].reset_index(),
            trade_df[["date", "trade balance"]],
            consump_df[["date", "current consumption"]],
            spot_df[["date", "CNYUSDSpot"]],
            neer_df[["date", "neer_yoy"]],
        ]
    ).sort_values("date").reset_index(drop=True)

    # Map quarterly GDP
    gdp_q = gdp_df[["date", "realgdp_chained"]].copy()
    gdp_q["date"] = gdp_q["date"].dt.to_timestamp(how="end").dt.to_period("M").dt.to_timestamp()
    gdp_monthly_df = gdp_monthly_df.merge(
        gdp_q.rename(columns={"realgdp_chained": "realgdp"}), on="date", how="left"
    )
    gdp_monthly_df = gdp_monthly_df[gdp_monthly_df["date"] < "2026-01-01"].reset_index(drop=True)
    return gdp_monthly_df


def save_gdp_construction_plot(gdp_monthly_df):
    """Save reported quarterly GDP vs proxy monthly GDP construction chart."""
    out_dir = PROJECT_ROOT / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly = gdp_monthly_df[["date", "realgdp_monthly"]].dropna().copy()
    quarterly = gdp_monthly_df[["date", "realgdp"]].dropna().copy()

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(
        monthly["date"],
        monthly["realgdp_monthly"],
        color="tab:blue",
        lw=1.6,
        label="Proxy monthly GDP level",
    )
    ax.scatter(
        quarterly["date"],
        quarterly["realgdp"] / 3.0,
        color="tab:red",
        s=22,
        zorder=3,
        label="Reported quarterly GDP / 3 (quarter-end)",
    )
    ax.set_title("Monthly GDP Construction: Reported Quarterly vs Proxy Monthly")
    ax.set_ylabel("Level (scaled)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()

    out_path = out_dir / "gdp_quarterly_vs_proxy_monthly.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("=" * 60)
    print("  prep_monthly_gdp: Building monthly GDP dataset")
    print("=" * 60)

    consump_df = load_consumption()
    trade_df, spot_df = load_trade()
    proxyfai_df = load_fai()
    gov_df = load_government()
    cpi_df = load_cpi()
    gdp_df = load_real_gdp()
    neer_df = load_neer()

    gdp_monthly_df = combine_monthly(cpi_df, gov_df, proxyfai_df, trade_df, consump_df, spot_df, neer_df, gdp_df)
    gdp_monthly_df = prepare_and_run(gdp_monthly_df, break_year=None)

    # Save intermediate result for downstream scripts
    gdp_monthly_df.to_csv(DERIVED_DIR / "gdp_monthly_df.csv", index=False)
    print(f"Saved: {DERIVED_DIR / 'gdp_monthly_df.csv'}")

    # Save appendix figure used in slides
    save_gdp_construction_plot(gdp_monthly_df)
    return gdp_monthly_df


if __name__ == "__main__":
    main()
