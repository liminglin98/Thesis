# Rule-Based Monetary Policy in China

This repository contains the empirical code for my master’s thesis project, **Rule-Based Monetary Policy in China: A Semi-Structural Counterfactual Analysis**.

The project studies how Chinese macroeconomic outcomes would have differed if the People’s Bank of China (PBoC) had followed explicit rule-based targets during the post-Covid low-inflation / deflation episode. The empirical workflow combines data construction, BVAR baseline projections, monetary policy shock identification, IV-SVAR impulse responses, and semi-structural counterfactual simulations.

## Research question

The central question is:

> What would have happened to China’s CPI inflation, real activity, exchange-rate conditions, and policy-rate path if the PBoC had followed stricter rule-based monetary policy targets rather than its historical discretionary policy stance?

The project is motivated by China’s recent difficulty in stabilizing the price level despite official growth and inflation objectives. The analysis asks whether missed inflation targets reflect insufficient monetary accommodation, weak monetary transmission, or broader structural constraints that cannot be resolved by an alternative policy rule alone.

## Methodological overview

The project uses a semi-structural counterfactual design. Rather than building a fully specified DSGE model for China, the current implementation relies on two empirical building blocks:

1. **A baseline projection model** for Chinese macroeconomic variables.
2. **Estimated causal effects of monetary policy shocks**, summarized by impulse response functions.

These objects are then combined to simulate counterfactual policy paths under alternative rules while holding non-monetary shocks fixed.

The current implementation uses two monetary policy shock designs:

- **Narrative / Romer-Romer-style shocks** recovered from a Chinese monetary policy reaction function.
- **High-frequency identification shocks** based on policy-event surprises in FR007-related market rates.

## Repository structure

The `reorganized` branch separates raw data, derived data, code, notebooks, and outputs:

```text
.
├── Data/
│   ├── raw/                         # Original data sources
│   └── derived/                     # Cleaned datasets produced by the Python pipeline
│       ├── gdp_monthly_df.csv
│       ├── china_longterm_data.csv
│       ├── romer_china_data.csv
│       └── hfi_core_data.csv
│
├── src/
│   ├── python/                      # Data preparation pipeline
│   │   ├── run_all.py               # Runs the full Python data pipeline
│   │   ├── config.py                # Shared path configuration
│   │   ├── fetch_akshare.py         # Optional data download utilities
│   │   ├── prep_monthly_gdp.py      # Monthly GDP interpolation
│   │   ├── prep_narrative.py        # Data for narrative shock identification
│   │   ├── prep_longterm.py         # Long-run macro panel for BVAR projection
│   │   ├── prep_hfi.py              # HFI shock dataset
│   │   └── stock_watson_distribute.py
│   │
│   └── julia/                       # Estimation and counterfactual analysis
│       ├── run_all.jl               # Runs the full Julia pipeline
│       ├── common.jl                # Shared VAR / IV / path utilities
│       ├── LTP.jl                   # Long-term projection BVAR
│       ├── RRShocks_monthly.jl      # Narrative shocks + BVAR/IV-SVAR IRFs
│       ├── HFIShocks.jl             # HFI shocks + BVAR/IV-SVAR IRFs
│       └── Counterfactual.jl        # Policy counterfactual simulations
│
├── notebooks/                       # Exploratory notebooks; not required for final pipeline
├── outputs/                         # Generated results, organized by sample and output type
├── Lin_Research Proposal.tex        # Research proposal
├── reference.bib                    # Bibliography
└── README.md
```

## Important path note

The reorganized tree currently stores data under `Data/`, but the centralized path configs in `src/python/config.py` and `src/julia/common.jl` point to lower-case `data/`. On a case-insensitive filesystem such as typical macOS setups, this may still work. On Linux or GitHub Actions, it will not. Before running the full pipeline on a case-sensitive filesystem, either rename `Data/` to `data/` or update the path constants in the config files.

## Quick start

From the repository root:

```bash
# 1. Run the data preparation pipeline
cd src/python
python run_all.py

# 2. Return to the repository root and run the Julia pipeline
cd ../..
julia src/julia/run_all.jl
```

The Python stage constructs the derived datasets. The Julia stage estimates the BVAR, constructs monetary policy IRFs, and runs the counterfactual simulations.

## Pipeline

### Stage 1: Data preparation in Python

`src/python/run_all.py` runs the data-preparation scripts in the following order:

| Step | Script | Main output | Purpose |
| --- | --- | --- | --- |
| 1 | `prep_monthly_gdp.py` | `gdp_monthly_df.csv` | Interpolates monthly GDP using Stock-Watson temporal distribution. |
| 2 | `prep_narrative.py` | `romer_china_data.csv` | Constructs the dataset for Romer-Romer-style narrative shock identification. |
| 3 | `prep_longterm.py` | `china_longterm_data.csv` | Builds the seven-variable monthly macro panel for the baseline BVAR. |
| 4 | `prep_hfi.py` | `hfi_core_data.csv` | Builds the HFI dataset with policy-event surprises and macro controls. |

### Stage 2: Estimation and analysis in Julia

`src/julia/run_all.jl` runs the Julia scripts in this order:

| Step | Script | Purpose | Main output |
| --- | --- | --- | --- |
| 1 | `LTP.jl` | Estimate the baseline BVAR with Minnesota prior. | `bvar_results.jls` |
| 2 | `RRShocks_monthly.jl` | Estimate narrative policy shocks and BVAR/IV-SVAR IRFs. | `narrative_irf_results.jls` |
| 3 | `HFIShocks.jl` | Estimate high-frequency identified shocks and BVAR/IV-SVAR IRFs. | `hfi_irf_ratechange.jls` |
| 4 | `Counterfactual.jl` | Simulate strict and flexible targeting counterfactuals. | `counterfactual_results.jls` |

The sample periods are defined centrally in `src/julia/common.jl`. The current branch defines two main samples:

| Label | Period | Use |
| --- | --- | --- |
| `2025` | 2002-01 to 2025-12 | Full-sample baseline. |
| `2022` | 2002-01 to 2022-12 | Pre-deflation / pre-late-sample robustness window. |

`common.jl` also defines segmented windows for shock-IRF diagnostics:

| Label | Period |
| --- | --- |
| `seg_2002_2015` | 2002-01 to 2015-12 |
| `seg_2015_2020` | 2015-01 to 2020-12 |
| `seg_2020_2025` | 2020-01 to 2025-12 |

## Baseline BVAR

`src/julia/LTP.jl` estimates a seven-variable monthly BVAR with a Minnesota prior. The current variable ordering is:

1. Real GDP growth, year-on-year
2. Industrial production growth, year-on-year
3. CPI inflation, year-on-year
4. FR007
5. M2 growth
6. NEER growth, year-on-year
7. US industrial production growth, year-on-year

The baseline specification uses six lags and computes Wold moving-average coefficients over a 120-month horizon. The resulting objects are saved by sample under:

```text
outputs/intermediate/{sample}/bvar_results.jls
```

## Monetary policy shock identification

### Narrative / Romer-Romer-style shocks

`src/julia/RRShocks_monthly.jl` estimates a Chinese monetary policy reaction function and interprets residual innovations as narrative monetary policy shocks. These shocks are then used in a BVAR/IV-SVAR framework to estimate impulse responses for the counterfactual exercise.

### High-frequency identification shocks

`src/julia/HFIShocks.jl` estimates high-frequency identified monetary policy shocks using policy-event surprises. The script estimates BVAR/IV-SVAR responses and saves the rate-change HFI IRF object used by the counterfactual analysis.

## Counterfactual analysis

`src/julia/Counterfactual.jl` combines:

- BVAR baseline projections,
- narrative monetary policy IRFs,
- HFI monetary policy IRFs, and
- alternative policy target rules.

The counterfactual exercise solves for the monetary policy shock sequence required to satisfy alternative policy rules, then compares the implied macro paths against the historical and baseline-projection paths.

The current counterfactual designs include:

1. **Strict CPI targeting** — force CPI inflation to match the target path.
2. **Strict government target rule** — use government CPI target paths, including the lower 2025 inflation target.
3. **Flexible inflation targeting** — minimize a weighted loss over inflation stabilization, output stabilization, interest-rate smoothing, and exchange-rate pressure.

## Output directories

Generated outputs are grouped by sample period:

```text
outputs/
├── intermediate/{2025,2022}/   # Serialized .jls objects
├── main_results/{2025,2022}/   # IRF plots and counterfactual figures
├── diagnostics/{2025,2022}/    # BVAR diagnostics and forecast checks
└── robustness/{2025,2022}/     # Robustness and alternative-specification figures
```

The repository is intended to track source code and essential data. Large generated figures, serialized objects, cache files, notebooks, and LaTeX build artifacts are excluded or should be regenerated locally.

## Dependencies

### Python

The data pipeline uses standard Python scientific-computing libraries. Depending on which data-fetching utilities are used, it may also require AkShare.

Typical dependencies include:

```bash
pip install pandas numpy scipy openpyxl akshare
```

### Julia

The Julia pipeline uses packages including:

- `CSV.jl`
- `DataFrames.jl`
- `Plots.jl`
- `GLM.jl`
- `StatsModels.jl`
- `ShiftedArrays.jl`
- `CovarianceMatrices.jl`
- `Distributions.jl`

A minimal installation command is:

```julia
using Pkg
Pkg.add([
    "CSV",
    "DataFrames",
    "Plots",
    "GLM",
    "StatsModels",
    "ShiftedArrays",
    "CovarianceMatrices",
    "Distributions"
])
```

## Current status

This repository is an active research workspace rather than a finalized replication package. The reorganized branch is designed to make the workflow cleaner and more reproducible by separating:

- raw inputs,
- derived datasets,
- Python data construction,
- Julia estimation code,
- exploratory notebooks, and
- generated outputs.

Remaining cleanup tasks include:

- resolving path-casing consistency between `Data/` and `data/`,
- adding Python and Julia environment files,
- documenting original data sources more explicitly,
- separating exploratory robustness checks from final thesis scripts,
- adding a one-command replication target,
- adding clearer versioning for thesis tables and figures.

## Author

**Liming Lin**  
Master’s thesis project, Sciences Po
