# Monetary Policy Transmission in China — BVAR / IV-SVAR / Counterfactual Analysis

## Quick Start

```bash
# 1. Data preparation (Python)
cd src/python && python run_all.py

# 2. Estimation & analysis (Julia)
julia src/julia/run_all.jl
```

## Repository Structure

```
.
├── Data/
│   ├── raw/                    # Raw data sources (CSV, XLSX)
│   └── derived/                # Cleaned datasets (produced by Python pipeline)
│       ├── gdp_monthly_df.csv
│       ├── china_longterm_data.csv
│       ├── romer_china_data.csv
│       └── hfi_core_data.csv
│
├── src/
│   ├── python/                 # Data preparation pipeline
│   │   ├── run_all.py          # Run all data prep steps
│   │   ├── config.py           # Shared paths & settings
│   │   ├── fetch_akshare.py    # Download data via AkShare API
│   │   ├── prep_monthly_gdp.py # Monthly GDP interpolation
│   │   ├── prep_narrative.py   # Narrative shock data (Romer-Romer style)
│   │   ├── prep_longterm.py    # Long-term macro panel (7 variables)
│   │   ├── prep_hfi.py         # High-frequency identification data
│   │   └── stock_watson_distribute.py  # Stock-Watson temporal distribution
│   │
│   └── julia/                  # Estimation & analysis
│       ├── run_all.jl          # Run full estimation pipeline
│       ├── common.jl           # Shared utilities (VAR helpers, IV identification)
│       ├── LTP.jl              # BVAR with Minnesota prior (7-var, 120-month horizon)
│       ├── RRShocks_monthly.jl # Narrative policy rule + BVAR+IV-SVAR (5-var)
│       ├── HFIShocks.jl        # HFI shocks + BVAR+IV-SVAR (5-var)
│       └── Counterfactual.jl   # Counterfactual scenarios (Wolf et al. 2025)
│
├── notebooks/                  # Exploratory Jupyter notebooks
│   ├── Data.ipynb
│   └── Data_Monthly.ipynb
│
├── outputs/                    # All estimation outputs (by sample period)
│   ├── intermediate/{2025,2019,2022}/  # Serialized .jls data
│   ├── main_results/{2025,2019,2022}/  # IRF plots & counterfactual figures
│   ├── diagnostics/{2025,2019,2022}/   # BVAR diagnostic plots
│   └── robustness/{2025,2019,2022}/    # Robustness checks
│
├── Literature Review/          # Literature notes
├── Lin_Research Proposal.tex   # Research proposal (LaTeX)
└── reference.bib               # Bibliography
```

## Pipeline

### Stage 1: Data Preparation (Python)

`src/python/run_all.py` runs the following in order:

| Script | Output | Description |
|--------|--------|-------------|
| `prep_monthly_gdp.py` | `gdp_monthly_df.csv` | Monthly GDP via Stock-Watson temporal distribution |
| `prep_narrative.py` | `romer_china_data.csv` | Policy rule variables + targets for narrative shock identification |
| `prep_longterm.py` | `china_longterm_data.csv` | 7-variable monthly macro panel (GDP, IP, CPI, FR007, M2, NEER, US IP) |
| `prep_hfi.py` | `hfi_core_data.csv` | 5-variable panel with daily FR007 surprises as HFI instruments |

### Stage 2: Estimation & Analysis (Julia)

`src/julia/run_all.jl` runs the following in order. Scripts 1-3 each loop over **3 sample periods**:

| Sample | Period | Purpose |
|--------|--------|---------|
| `2025` | 2002-01 to 2025-12 | Baseline (full sample) |
| `2019` | 2002-01 to 2019-12 | Pre-COVID |
| `2022` | 2002-01 to 2022-12 | Pre-deflation |

| Script | Description | Key Output |
|--------|-------------|------------|
| `LTP.jl` | BVAR with Minnesota prior (7-var, p=6, H=120) | `bvar_results.jls` |
| `RRShocks_monthly.jl` | Narrative (Romer-Romer) policy rule + IV-SVAR identification | `narrative_irf_results.jls` |
| `HFIShocks.jl` | High-frequency identified shocks + IV-SVAR | `hfi_irf_ratechange.jls` |
| `Counterfactual.jl` | Strict & flexible inflation targeting scenarios (Wolf et al. 2025) | `counterfactual_results.jls` |

Sample periods are defined in `common.jl` (`SAMPLES` constant) and can be edited in one place.

### Output Categories

Each category is organized by sample year (`2025/`, `2019/`, `2022/`):

| Folder | Contents |
|--------|----------|
| `outputs/intermediate/{year}/` | Serialized `.jls` files — BVAR coefficients, IRF posterior draws, counterfactual results |
| `outputs/main_results/{year}/` | IRF plots (narrative, HFI, comparison) and counterfactual scenario figures |
| `outputs/diagnostics/{year}/` | BVAR residual plots, Wold coefficient decay, GDP in-sample fit & forecast |
| `outputs/robustness/{year}/` | Policy rule comparison (full sample vs excl. 2020), alternative HFI instrument IRFs |
