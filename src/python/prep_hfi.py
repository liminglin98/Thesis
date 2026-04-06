"""
High-frequency identification data preparation.
Cells 67–86 of the original Data_Monthly notebook.
Output: hfi_core_data.csv
"""

import re
import pandas as pd
from functools import reduce
from config import RAW_DIR, DERIVED_DIR
from fetch_akshare import fetch_fr007_daily


# Sheet-to-column rename maps for CMPI.xlsx
SHEET_RENAMES = {
    "正回购利率": {"Unnamed: 1": "Repo14days", "Unnamed: 2": "Repo21days", "Unnamed: 3": "Repo28days",
                    "Unnamed: 4": "Repo91days", "Unnamed: 5": "Repo182days", "Unnamed: 6": "Repo364days"},
    "逆回购利率": {"Unnamed: 1": "ReRepo7days", "Unnamed: 2": "ReRepo14days", "Unnamed: 3": "ReRepo21days",
                    "Unnamed: 4": "ReRepo28days", "Unnamed: 5": "ReRepo91days"},
    "国库现金利率": {"Unnamed: 1": "TC3months", "Unnamed: 2": "TC6months", "Unnamed: 3": "TC9months"},
    "常备借贷便利利率": {"Unnamed: 1": "SLFovernight", "Unnamed: 2": "SLF7days", "Unnamed: 3": "SLF1month"},
    "短期流动性工具利率": {"表五：短期流动性调节工具（SLO）操作利率": "SLO", "Unnamed: 2": "ReverseSLO"},
    "央行票据发行利率": {"Unnamed: 1": "CBB3months", "Unnamed: 2": "CBB6months", "Unnamed: 3": "CBB1year",
                          "Unnamed: 4": "CBB3years"},
    "定期存款基准利率": {"Unnamed: 1": "TD3months", "Unnamed: 2": "TD1year"},
    "金融机构在央行存款利率": {"Unnamed: 1": "RequiredReserve", "Unnamed: 2": "ExcessReserve"},
    "中长期贷款基准利率": {"Unnamed: 1": "ML1-3years", "Unnamed: 2": "ML3-5years"},
    "中期借贷便利MLF利率": {"Unnamed: 1": "MLF3months", "Unnamed: 2": "MLF6months", "Unnamed: 3": "MLF1year"},
    "抵押补充贷款利率": {"Unnamed: 1": "PSL"},
    "贷款市场报价利率LPR": {"Unnamed: 1": "LPR1year", "Unnamed: 2": "LPR5years"},
    "存款准备金率RRR": {"Unnamed: 1": "RRRLarge", "Unnamed: 2": "RRRSmall"},
}


def parse_cmpi_sheets():
    """Read all sheets from CMPI.xlsx, rename columns, return dict of DataFrames."""
    excel_file = pd.ExcelFile(RAW_DIR / "CMPI.xlsx")
    cmpi_dfs = {}
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(RAW_DIR / "CMPI.xlsx", sheet_name=sheet_name)
        df = df.iloc[1:].copy()
        first_col = df.columns[0]
        df.rename(columns={first_col: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].dt.year >= 2000]

        if sheet_name in SHEET_RENAMES:
            df.rename(columns=SHEET_RENAMES[sheet_name], inplace=True)

        clean_name = re.sub(r"[（）()]", "", sheet_name).strip()
        cmpi_dfs[clean_name] = df
    return cmpi_dfs


def build_hfi_matrix(cmpi_dfs):
    """Build daily binary announcement matrix from all CMPI sheets."""
    binary_series = []
    for df in cmpi_dfs.values():
        data_cols = [c for c in df.columns if c != "date"]
        bin_df = df[["date"]].copy()
        for col in data_cols:
            bin_df[col] = df[col].notna().astype(int)
        binary_series.append(bin_df)

    hfi_df = reduce(lambda l, r: pd.merge(l, r, on="date", how="outer"), binary_series)
    date_range = pd.DataFrame({"date": pd.date_range("2000-01-01", hfi_df["date"].max(), freq="D")})
    hfi_df = date_range.merge(hfi_df, on="date", how="left").fillna(0)

    instrument_cols = [c for c in hfi_df.columns if c != "date"]
    hfi_df[instrument_cols] = hfi_df[instrument_cols].astype(int)
    hfi_df = hfi_df.sort_values("date").reset_index(drop=True)
    hfi_df["any_policy"] = hfi_df[instrument_cols].any(axis=1).astype(int)
    return hfi_df, instrument_cols


def select_core_instruments(hfi_df, cmpi_dfs):
    """Select core policy instruments and detect rate changes."""
    selected_cols = ["date", "ReRepo7days", "MLF1year", "LPR1year", "RRRLarge", "TD1year", "ML1-3years"]
    hfi_core_df = hfi_df[selected_cols].copy()
    core_cols = [c for c in hfi_core_df.columns if c != "date"]
    hfi_core_df["any_policy"] = hfi_core_df[core_cols].any(axis=1).astype(int)

    # Detect rate changes using source DataFrames
    core_map = {
        "ReRepo7days": cmpi_dfs.get("逆回购利率"),
        "MLF1year": cmpi_dfs.get("中期借贷便利MLF利率"),
        "LPR1year": cmpi_dfs.get("贷款市场报价利率LPR"),
        "RRRLarge": cmpi_dfs.get("存款准备金率RRR"),
        "TD1year": cmpi_dfs.get("定期存款基准利率"),
        "ML1-3years": cmpi_dfs.get("中长期贷款基准利率"),
    }

    change_flags = []
    for col, src_df in core_map.items():
        if src_df is None or col not in src_df.columns:
            continue
        vals = src_df[["date", col]].dropna().sort_values("date").copy()
        vals["_prev"] = vals[col].shift(1)
        vals[f"{col}_changed"] = (vals["_prev"].notna() & (vals[col] != vals["_prev"])).astype(int)
        change_flags.append(vals[["date", f"{col}_changed"]])

    change_df = change_flags[0]
    for cf in change_flags[1:]:
        change_df = change_df.merge(cf, on="date", how="outer")

    hfi_core_df = hfi_core_df.merge(change_df, on="date", how="left")
    change_col_names = [f"{c}_changed" for c in core_map if f"{c}_changed" in hfi_core_df.columns]
    hfi_core_df[change_col_names] = hfi_core_df[change_col_names].fillna(0).astype(int)
    hfi_core_df["any_policy_change"] = hfi_core_df[change_col_names].any(axis=1).astype(int)
    hfi_core_df = hfi_core_df.drop(columns=change_col_names)
    return hfi_core_df


def map_to_fr007(hfi_core_df):
    """Map HFI events to daily FR007 changes, then aggregate monthly."""
    daily_repo_df = fetch_fr007_daily()
    daily_repo_df["date"] = pd.to_datetime(daily_repo_df["date"])
    daily_repo_df = daily_repo_df[["date", "FR007"]].copy()

    shock_df = hfi_core_df.merge(daily_repo_df, on="date", how="left")
    shock_df["FR007_prev"] = shock_df["FR007"].shift(1)
    shock_df["FR007_diff"] = shock_df["FR007"] - shock_df["FR007_prev"]
    shock_df["shock_policy"] = shock_df["FR007_diff"].where(shock_df["any_policy"] == 1).fillna(0)
    shock_df["shock_policy_change"] = shock_df["FR007_diff"].where(shock_df["any_policy_change"] == 1).fillna(0)

    # Monthly aggregation
    monthly = (
        shock_df[["date", "shock_policy", "shock_policy_change"]]
        .groupby(shock_df["date"].dt.to_period("M"))
        .agg({"shock_policy": "sum", "shock_policy_change": "sum"})
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()
    return monthly


def main():
    print("=" * 60)
    print("  prep_hfi: Building HFI shocks dataset")
    print("=" * 60)

    cmpi_dfs = parse_cmpi_sheets()
    hfi_df, instrument_cols = build_hfi_matrix(cmpi_dfs)
    hfi_core_df = select_core_instruments(hfi_df, cmpi_dfs)
    monthly_shocks = map_to_fr007(hfi_core_df)

    # Merge with romer data
    romer_df = pd.read_csv(DERIVED_DIR / "romer_china_data.csv", parse_dates=["date"])
    hfi_core_csv_df = romer_df.merge(monthly_shocks, on="date", how="left")
    hfi_core_csv_df.to_csv(DERIVED_DIR / "hfi_core_data.csv", index=False)
    print(f"Saved: {DERIVED_DIR / 'hfi_core_data.csv'}")
    return hfi_core_csv_df


if __name__ == "__main__":
    main()
