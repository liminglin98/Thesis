"""
Narrative shocks data preparation.
Cells 41–56 of the original Data_Monthly notebook.
Output: romer_china_data.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from config import PROJECT_ROOT, RAW_DIR, DERIVED_DIR
from fetch_akshare import fetch_fr007_daily, fetch_cpr_daily


def load_fr007_monthly():
    """Fetch daily FR007, collapse to monthly mean."""
    full_repo_df = fetch_fr007_daily()
    full_repo_df["date"] = pd.to_datetime(full_repo_df["date"])
    full_repo_df = (
        full_repo_df[["date", "FR007"]]
        .groupby(full_repo_df["date"].dt.to_period("M"))["FR007"]
        .mean()
        .reset_index()
    )
    full_repo_df["date"] = full_repo_df["date"].dt.to_timestamp()
    return full_repo_df


def load_official_rates():
    """Official OMO 7-day reverse repo rate from CMPI.xlsx."""
    omo_df = pd.read_excel(RAW_DIR / "CMPI.xlsx", sheet_name="逆回购利率", header=1, index_col=False)
    omo_df = omo_df[["时间", "7天"]].rename(columns={"时间": "date", "7天": "omo7d"})
    omo_df["date"] = pd.to_datetime(omo_df["date"], errors="coerce")
    omo_df["omo7d"] = pd.to_numeric(omo_df["omo7d"], errors="coerce")
    omo_df = omo_df.sort_values("date").ffill()
    omo_df = omo_df[omo_df["date"].dt.year >= 2000]

    omo_df = (
        omo_df.groupby(omo_df["date"].dt.to_period("M"))["omo7d"]
        .mean()
        .reset_index()
    )
    omo_df["date"] = omo_df["date"].dt.to_timestamp()

    full_months = pd.date_range(omo_df["date"].min(), omo_df["date"].max(), freq="MS")
    omo_df = omo_df.set_index("date").reindex(full_months).ffill().rename_axis("date").reset_index()
    return omo_df


def load_central_parity():
    """Central parity rate (USD/CNY), monthly mean."""
    cpr_df = fetch_cpr_daily()
    cpr_df["date"] = pd.to_datetime(cpr_df["date"])
    cpr_df = cpr_df[cpr_df["date"].dt.year >= 2000]
    cpr_df = (
        cpr_df.groupby(cpr_df["date"].dt.to_period("M"))["CNYUSDCPR"]
        .mean()
        .reset_index()
    )
    cpr_df["CNYUSDCPR"] = cpr_df["CNYUSDCPR"] / 100
    cpr_df["date"] = cpr_df["date"].dt.to_timestamp()
    return cpr_df


def load_industrial_output():
    """Industrial production YoY and cumulative."""
    ip_df = pd.read_csv(
        RAW_DIR / "IP.csv",
        encoding="gbk", sep=",", header=2, index_col=0, engine="python",
    )
    ip_df = ip_df.iloc[:2, :].copy().T.rename_axis("date").reset_index()
    ip_df["date"] = pd.to_datetime(ip_df["date"], format="%Y年%m月", errors="coerce")
    ip_df.columns = ["date", "IP_yoy", "IP_cumulative"]

    year = ip_df["date"].dt.year
    month = ip_df["date"].dt.month
    feb_cum = ip_df.loc[month.eq(2)].set_index(year[month.eq(2)])["IP_cumulative"]
    for m in [1, 2]:
        mask = month.eq(m) & ip_df["IP_yoy"].isna()
        ip_df.loc[mask, "IP_yoy"] = year[mask].map(feb_cum)
    return ip_df


def load_targets(romer_df):
    """Map official GDP and CPI targets to the dataframe."""
    china_gdp_targets = {
        2000: 7.0, 2001: 7.0, 2002: 7.0, 2003: 7.0, 2004: 7.0,
        2005: 8.0, 2006: 8.0, 2007: 8.0, 2008: 8.0, 2009: 8.0,
        2010: 8.0, 2011: 8.0, 2012: 7.5, 2013: 7.5, 2014: 7.5,
        2015: 7.0, 2016: 6.75, 2017: 6.5, 2018: 6.5, 2019: 6.25,
        2020: 6.0, 2021: 6.0, 2022: 5.5, 2023: 5.0, 2024: 5.0, 2025: 5.0,
    }
    china_cpi_targets = {
        2000: 2.0, 2001: 1.5, 2002: 1.5, 2003: 1.0, 2004: 3.0,
        2005: 4.0, 2006: 3.0, 2007: 3.0, 2008: 4.8, 2009: 4.0,
        2010: 3.0, 2011: 4.0, 2012: 4.0, 2013: 3.5, 2014: 3.5,
        2015: 3.0, 2016: 3.0, 2017: 3.0, 2018: 3.0, 2019: 3.0,
        2020: 3.5, 2021: 3.0, 2022: 3.0, 2023: 3.0, 2024: 3.0, 2025: 2.0,
    }
    years = romer_df["date"].dt.year
    romer_df["target_gdp"] = years.map(china_gdp_targets)
    romer_df["target_cpi"] = years.map(china_cpi_targets)
    return romer_df


def _parse_cmpi_sheet(sheet_df):
    """Parse one CMPI sheet into date + sheet average rate."""
    df = sheet_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    first_col = df.columns[0]
    df["date"] = pd.to_datetime(df[first_col], errors="coerce")

    if df["date"].notna().sum() < 5 and len(df) > 1:
        alt = sheet_df.copy()
        alt.columns = [str(x).strip() for x in alt.iloc[0].tolist()]
        alt = alt.iloc[1:].copy()
        alt_first = alt.columns[0]
        alt["date"] = pd.to_datetime(alt[alt_first], errors="coerce")
        df = alt

    numeric_cols = [c for c in df.columns if c != "date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sheet_avg"] = df[numeric_cols].mean(axis=1, skipna=True)
    return df[["date", "sheet_avg"]].dropna()


def save_cmpi_vs_fr007_plot(romer_df):
    """Save standardized CMPI composite proxy vs FR007 monthly comparison chart."""
    out_dir = PROJECT_ROOT / "outputs" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmfile = RAW_DIR / "CMPI.xlsx"
    xls = pd.ExcelFile(cmfile)

    parsed = []
    for sheet_name in xls.sheet_names:
        sheet_df = pd.read_excel(cmfile, sheet_name=sheet_name)
        parsed_sheet = _parse_cmpi_sheet(sheet_df)
        if not parsed_sheet.empty:
            parsed.append(parsed_sheet)

    if not parsed:
        print("Warning: no CMPI series parsed; skipped CMPI vs FR007 plot")
        return

    cmpi_df = pd.concat(parsed, ignore_index=True)
    cmpi_df = (
        cmpi_df.groupby(pd.Grouper(key="date", freq="ME"))["sheet_avg"]
        .mean()
        .reset_index()
        .rename(columns={"sheet_avg": "cmpi_proxy"})
    )

    plot_df = romer_df[["date", "FR007"]].dropna().merge(cmpi_df, on="date", how="inner")
    plot_df = plot_df.dropna(subset=["FR007", "cmpi_proxy"]).copy()
    if plot_df.empty:
        print("Warning: CMPI merge produced empty data; skipped CMPI vs FR007 plot")
        return

    for col in ["FR007", "cmpi_proxy"]:
        std = plot_df[col].std(ddof=0)
        plot_df[f"{col}_z"] = (plot_df[col] - plot_df[col].mean()) / (std if std else 1.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(plot_df["date"], plot_df["FR007_z"], color="tab:blue", lw=1.7, label="FR007 (z-score)")
    ax.plot(
        plot_df["date"],
        plot_df["cmpi_proxy_z"],
        color="tab:orange",
        lw=1.7,
        label="CMPI composite proxy (z-score)",
    )
    ax.set_title("CMPI vs FR007 (Standardized Levels)")
    ax.set_ylabel("Standardized units")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()

    out_path = out_dir / "cmpi_vs_fr007.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("=" * 60)
    print("  prep_narrative: Building narrative shocks dataset")
    print("=" * 60)

    # Load gdp_monthly_df from upstream
    gdp_monthly_df = pd.read_csv(DERIVED_DIR / "gdp_monthly_df.csv", parse_dates=["date"])

    full_repo_df = load_fr007_monthly()
    omo_df = load_official_rates()
    cpr_df = load_central_parity()
    ip_df = load_industrial_output()
    neer_df = pd.read_csv(
        RAW_DIR / "RBCNBIS_PC1.csv", encoding="gbk", sep=",", header=0, index_col=0, engine="python"
    )
    neer_df = neer_df.reset_index().rename(columns={"observation_date": "date", "RBCNBIS_PC1": "neer_yoy"})
    neer_df["date"] = pd.to_datetime(neer_df["date"], errors="coerce")

    romer_df = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        [
            gdp_monthly_df[["date", "realgdp_monthly_yoy", "cpi", "CNYUSDSpot", "trade balance", "current consumption"]],
            full_repo_df[["date", "FR007"]],
            omo_df[["date", "omo7d"]],
            cpr_df[["date", "CNYUSDCPR"]],
            ip_df[["date", "IP_yoy"]],
            neer_df[["date", "neer_yoy"]],
        ]
    ).sort_values("date").reset_index(drop=True)

    romer_df = load_targets(romer_df)
    romer_df.to_csv(DERIVED_DIR / "romer_china_data.csv", index=False)
    print(f"Saved: {DERIVED_DIR / 'romer_china_data.csv'}")

    # Save appendix figure used in slides
    save_cmpi_vs_fr007_plot(romer_df)
    return romer_df


if __name__ == "__main__":
    main()
