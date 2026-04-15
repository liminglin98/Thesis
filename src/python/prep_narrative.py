"""
Narrative shocks data preparation.
Cells 41–56 of the original Data_Monthly notebook.
Output: romer_china_data.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
from functools import reduce
from config import PROJECT_ROOT, RAW_DIR, DERIVED_DIR
from fetch_akshare import fetch_fr007_daily, fetch_cpr_daily


CMPI_RENAME_MAP = {
    "正回购利率": {
        "Unnamed: 1": "Repo14days",
        "Unnamed: 2": "Repo21days",
        "Unnamed: 3": "Repo28days",
        "Unnamed: 4": "Repo91days",
        "Unnamed: 5": "Repo182days",
        "Unnamed: 6": "Repo364days",
    },
    "逆回购利率": {
        "Unnamed: 1": "ReRepo7days",
        "Unnamed: 2": "ReRepo14days",
        "Unnamed: 3": "ReRepo21days",
        "Unnamed: 4": "ReRepo28days",
        "Unnamed: 5": "ReRepo91days",
    },
    "国库现金利率": {
        "Unnamed: 1": "TC3months",
        "Unnamed: 2": "TC6months",
        "Unnamed: 3": "TC9months",
    },
    "常备借贷便利利率": {
        "Unnamed: 1": "SLFovernight",
        "Unnamed: 2": "SLF7days",
        "Unnamed: 3": "SLF1month",
    },
    "短期流动性工具利率": {
        "表五：短期流动性调节工具（SLO）操作利率": "SLO",
        "Unnamed: 2": "ReverseSLO",
    },
    "央行票据发行利率": {
        "Unnamed: 1": "CBB3months",
        "Unnamed: 2": "CBB6months",
        "Unnamed: 3": "CBB1year",
        "Unnamed: 4": "CBB3years",
    },
    "定期存款基准利率": {
        "Unnamed: 1": "TD3months",
        "Unnamed: 2": "TD1year",
    },
    "金融机构在央行存款利率": {
        "Unnamed: 1": "RequiredReserve",
        "Unnamed: 2": "ExcessReserve",
    },
    "中长期贷款基准利率": {
        "Unnamed: 1": "ML1_3years",
        "Unnamed: 2": "ML3_5years",
    },
    "中期借贷便利MLF利率": {
        "Unnamed: 1": "MLF3months",
        "Unnamed: 2": "MLF6months",
        "Unnamed: 3": "MLF1year",
    },
    "抵押补充贷款利率": {
        "Unnamed: 1": "PSL",
    },
    "贷款市场报价利率LPR": {
        "Unnamed: 1": "LPR1year",
        "Unnamed: 2": "LPR5years",
    },
    "存款准备金率RRR": {
        "Unnamed: 1": "RRRLarge",
        "Unnamed: 2": "RRRSmall",
    },
}


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style*")
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
    """Map official GDP and CPI targets, with new annual targets taking effect in March."""
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
    announced_year = np.where(
        romer_df["date"].dt.month >= 3,
        romer_df["date"].dt.year,
        romer_df["date"].dt.year - 1,
    )
    announced_year = pd.Series(announced_year, index=romer_df.index)

    # If the previous year's target is unavailable at the sample start, fall back to same-year target.
    romer_df["target_gdp"] = announced_year.map(china_gdp_targets)
    romer_df["target_gdp"] = romer_df["target_gdp"].fillna(romer_df["date"].dt.year.map(china_gdp_targets))

    romer_df["target_cpi"] = announced_year.map(china_cpi_targets)
    romer_df["target_cpi"] = romer_df["target_cpi"].fillna(romer_df["date"].dt.year.map(china_cpi_targets))
    return romer_df


def load_cmpi_monthly():
    """Build monthly CMPI using the same logic as Data.ipynb, with month aggregation."""
    cmfile = RAW_DIR / "CMPI.xlsx"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style*")
        xls = pd.ExcelFile(cmfile)

    cmpi_dfs = {}
    for sheet_name in xls.sheet_names:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Workbook contains no default style*")
            df = pd.read_excel(cmfile, sheet_name=sheet_name)

        df = df.iloc[1:].copy()
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "date"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].dt.year >= 2000]

        clean_sheet_name = re.sub(r"[（）()]", "", sheet_name).strip()
        rename_dict = CMPI_RENAME_MAP.get(clean_sheet_name, {})
        if rename_dict:
            df = df.rename(columns=rename_dict)

        df["month"] = df["date"].dt.to_period("M")
        candidate_cols = [c for c in df.columns if c not in ["date", "month"]]
        for col in candidate_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            sheet_prefix = re.sub(r"\s+", "", clean_sheet_name)
            prefixed_map = {col: f"{sheet_prefix}_{col}" for col in numeric_cols}
            df = df[["month"] + numeric_cols].rename(columns=prefixed_map)
            df = df.groupby("month", as_index=False).mean(numeric_only=True)
        else:
            df = df[["month"]].drop_duplicates().reset_index(drop=True)

        cmpi_dfs[f"{clean_sheet_name}_df"] = df

    if not cmpi_dfs:
        return pd.DataFrame(columns=["date", "cmpi", "cmpi_n_indicators"])

    cmpi_df = reduce(
        lambda left, right: pd.merge(left, right, on="month", how="outer"),
        list(cmpi_dfs.values())
    )

    numeric_cols = cmpi_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        mean_val = cmpi_df[col].mean()
        std_val = cmpi_df[col].std()
        if pd.notna(std_val) and std_val != 0:
            cmpi_df[f"{col}_standardized"] = (cmpi_df[col] - mean_val) / std_val
        else:
            cmpi_df[f"{col}_standardized"] = np.nan

    standardized_cols = [c for c in cmpi_df.columns if c.endswith("_standardized")]
    cmpi_df["cmpi"] = cmpi_df[standardized_cols].mean(axis=1)
    cmpi_df["cmpi_n_indicators"] = cmpi_df[standardized_cols].notna().sum(axis=1)
    cmpi_df["date"] = cmpi_df["month"].dt.to_timestamp()

    return cmpi_df[["date", "cmpi", "cmpi_n_indicators"]]


def _trend_deviation(series, window=12):
    """Return the deviation of a monthly series from a smooth trend."""
    trend = series.rolling(window=window, center=True, min_periods=6).mean()
    trend = trend.bfill().ffill()
    return series - trend


def save_cmpi_vs_fr007_plot(romer_df):
    """Save monthly CMPI vs FR007 trend-deviation comparison chart."""
    out_dir = PROJECT_ROOT / "outputs" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "cmpi" not in romer_df.columns:
        cmpi_df = load_cmpi_monthly()
        plot_df = romer_df[["date", "FR007"]].merge(cmpi_df[["date", "cmpi"]], on="date", how="left")
    else:
        plot_df = romer_df[["date", "FR007", "cmpi"]].copy()

    plot_df["date"] = pd.to_datetime(plot_df["date"]).dt.to_period("M").dt.to_timestamp()
    plot_df = plot_df.dropna(subset=["FR007", "cmpi"]).copy()
    if plot_df.empty:
        print("Warning: CMPI merge produced empty data; skipped CMPI vs FR007 plot")
        return

    for col in ["FR007", "cmpi"]:
        plot_df[f"{col}_trend_dev"] = _trend_deviation(plot_df[col])
        std = plot_df[f"{col}_trend_dev"].std(ddof=0)
        plot_df[f"{col}_trend_dev_z"] = (
            plot_df[f"{col}_trend_dev"] - plot_df[f"{col}_trend_dev"].mean()
        ) / (std if std else 1.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(plot_df["date"], plot_df["FR007_trend_dev_z"], color="tab:blue", lw=1.7, label="FR007 (trend deviation)")
    ax.plot(
        plot_df["date"],
        plot_df["cmpi_trend_dev_z"],
        color="tab:orange",
        lw=1.7,
        label="CMPI (trend deviation)",
    )
    ax.set_title("CMPI vs FR007 (Monthly Deviations from Trend)")
    ax.set_ylabel("Standardized trend deviation")
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
    cmpi_df = load_cmpi_monthly()
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
            cmpi_df[["date", "cmpi", "cmpi_n_indicators"]],
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
