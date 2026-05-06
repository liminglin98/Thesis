# Rule-Based Monetary Policy in China

This repository contains the empirical and computational code for my master’s thesis project, **Rule-Based Monetary Policy in China: A Semi-Structural Counterfactual Analysis**.

The project asks how China’s macroeconomic outcomes would have differed if the People’s Bank of China (PBoC) had followed explicit rule-based monetary policy targets during the recent low-inflation and deflationary episode. The current implementation combines monetary policy shock identification, BVAR/IV-SVAR impulse responses, and semi-structural counterfactual simulations.

## Research question

The central question is:

> What would have happened to Chinese CPI inflation, real activity, and policy rates if the PBoC had followed stricter target rules instead of its historical discretionary policy path?

The thesis focuses especially on the post-Covid period, when China experienced near-zero CPI inflation, persistent PPI deflation, real-estate-sector stress, and concerns about weak monetary transmission.

## Methodological overview

The repository implements a semi-structural counterfactual design inspired by the policy-counterfactual literature. The approach uses:

1. **A baseline projection model** for Chinese macroeconomic variables.
2. **Estimated causal effects of monetary policy shocks**, represented by impulse response functions.
3. **Counterfactual policy rules**, such as strict CPI targeting or government target rules.
4. **Simulated counterfactual paths** for CPI, GDP, industrial production, and the policy rate, holding non-monetary shocks fixed.

The empirical strategy currently uses two monetary policy shock series:

- **Narrative / Romer-Romer-style shocks** estimated from a Chinese monetary policy reaction function.
- **High-frequency identification shocks** based on policy-event surprises in market rates.

These shock series are then used to estimate the dynamic effects of monetary policy and to construct counterfactual policy paths.

## Repository structure

```text
.
├── README.md
├── Lin_Research Proposal.tex
├── reference.bib
└── Data/
    ├── Data.ipynb
    ├── LTP.jl
    ├── RRShocks.jl
    ├── RRShocks_monthly.jl
    ├── HFIShocks.jl
    ├── Counterfactual.jl
    ├── *.csv
    ├── *.jls
    └── *.png
```

## Main files

| File | Purpose |
| --- | --- |
| `Lin_Research Proposal.tex` | Research proposal describing the motivation, research question, methodology, assumptions, and preliminary results. |
| `reference.bib` | Bibliography for the thesis project. |
| `Data/Data.ipynb` | Data-cleaning notebook used to construct the merged datasets used by the Julia scripts. |
| `Data/LTP.jl` | Estimates the baseline monthly BVAR with a Minnesota prior and saves BVAR objects for counterfactual analysis. |
| `Data/RRShocks.jl` | Constructs a Romer-Romer-style monetary policy shock series and estimates quarterly SVAR impulse responses. |
| `Data/RRShocks_monthly.jl` | Monthly version of the narrative shock / IRF workflow used by the counterfactual script. |
| `Data/HFIShocks.jl` | Estimates high-frequency-identification monetary policy shocks and BVAR+IV-SVAR impulse responses. |
| `Data/Counterfactual.jl` | Combines BVAR baseline projections and monetary policy IRFs to simulate strict and flexible targeting counterfactuals. |

## Baseline projection model

`Data/LTP.jl` estimates a seven-variable monthly BVAR with a Minnesota prior. The current variable ordering is:

1. Real GDP growth, year-on-year
2. Industrial production growth, year-on-year
3. CPI inflation, year-on-year
4. FR007 policy rate
5. M2 growth
6. NEER growth, year-on-year
7. US industrial production growth, year-on-year

The script uses six lags and computes Wold moving-average coefficients over a 120-month horizon. It saves the estimated model objects to `bvar_results.jls`, which is then loaded by the counterfactual script.

## Monetary policy shocks

### Narrative / Romer-Romer-style shocks

`Data/RRShocks.jl` estimates a Chinese monetary policy reaction function in the spirit of Romer and Romer. The policy residual is interpreted as a monetary policy shock. The reaction function includes lagged policy stance, inflation-gap terms, output-gap terms, asymmetric output-gap responses, and an exchange-rate-gap term.

The script then estimates impulse responses using an SVAR framework and plots responses of real activity, inflation, the monetary policy index, FR007, and the exchange rate.

### High-frequency-identification shocks

`Data/HFIShocks.jl` estimates monetary policy shocks using high-frequency surprises around policy events. It supports two instruments:

- `shock_policy`: any policy-announcement event.
- `shock_policy_change`: rate-change-only events.

The script estimates BVAR+IV-SVAR impulse responses, reports first-stage diagnostics, constructs posterior credible sets, and saves the rate-change IRF object for the counterfactual analysis.

## Counterfactual analysis

`Data/Counterfactual.jl` implements the main policy-counterfactual exercise. It loads:

- baseline BVAR objects from `bvar_results.jls`,
- narrative IRFs from `narrative_irf_results.jls`, and
- HFI IRFs from `hfi_irf_ratechange.jls`.

It then constructs policy transmission maps and solves for the monetary policy shock sequence needed to satisfy alternative policy rules.

The current counterfactuals include:

1. **Strict CPI targeting**: force CPI inflation to match the target path.
2. **Strict government target rule**: use government CPI targets, with 3% in earlier years and 2% in 2025.
3. **Flexible inflation targeting**: minimize a weighted loss over inflation stabilization, output stabilization, interest-rate smoothing, and exchange-rate pressure.

The script currently evaluates counterfactual windows beginning in 2021 and 2023 and produces plots comparing actual, baseline, and counterfactual paths.

## Expected workflow

Run the files from inside the `Data/` directory.

```bash
cd Data
```

A typical workflow is:

```julia
include("Data.jl")                 # if converted from the notebook, optional
include("LTP.jl")
include("RRShocks_monthly.jl")
include("HFIShocks.jl")
include("Counterfactual.jl")
```

At present, the data-cleaning step is mainly contained in `Data.ipynb`, so users should first run the notebook to make sure the intermediate CSV files exist.

## Dependencies

The project is written primarily in Julia, with some Python used for data processing. The Julia scripts use packages including:

- `CSV.jl`
- `DataFrames.jl`
- `Dates`
- `Statistics`
- `LinearAlgebra`
- `Serialization`
- `Plots.jl`
- `GLM.jl`
- `StatsModels.jl`
- `ShiftedArrays.jl`
- `CovarianceMatrices.jl`
- `Distributions.jl`
- `Random`
- `Printf`

To install the main Julia dependencies:

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

## Outputs

The scripts generate intermediate serialized objects and figures, including:

- BVAR objects: `bvar_results.jls`
- Narrative IRFs: `narrative_irf_results.jls`
- HFI IRFs: `hfi_irf_ratechange.jls`
- BVAR diagnostics and forecasts: `bvar_*.png`
- Monetary policy rule plots: `CMPI Comparison*.png`
- SVAR / IV-SVAR IRFs: `irf_*.png`
- Counterfactual plots: `cnfctl_*.png`

## Current status

This repository is an active research workspace rather than a polished replication package. The code is being used to develop and test the thesis methodology, so file paths, input datasets, and serialized intermediate objects may change.

Planned improvements include:

- consolidating the data-cleaning workflow into a reproducible script,
- adding a Julia `Project.toml` / `Manifest.toml`,
- documenting data sources and transformations more explicitly,
- separating exploratory code from final replication scripts,
- adding robustness checks for alternative shock definitions and policy instruments,
- improving the treatment of uncertainty in the counterfactual simulations.

## Thesis context

The thesis is motivated by China’s recent difficulty in stabilizing the price level despite official growth and inflation objectives. The project asks whether the observed failure to meet inflation targets reflects insufficient monetary accommodation, weak policy transmission, or broader structural constraints that cannot be resolved by a different policy rule alone.

## Author

**Liming Lin**  
Master’s thesis project, Sciences Po
