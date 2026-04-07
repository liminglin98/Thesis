##
# =============================================================================
# COUNTERFACTUAL HISTORICAL SCENARIO: STRICT INFLATION TARGETING
# =============================================================================
#
# Following Wolf et al. (2025) — Proposition 1 & hist_scenario application
#
# Question: What would have happened to the Chinese economy if the PBoC had
# strictly targeted CPI = 2% YoY from January 2023 onward?
#
# Method:
#   (1) BVAR baseline projection from 2023:01 (from LTP.jl)
#   (2) Monetary policy IRFs (from RRShocks_monthly.jl / HFIShocks.jl)
#   (3) Counterfactual shock sequence: solve for ν̃ such that the counterfactual
#       CPI path = 2% (the government work report target)
#
# The counterfactual modifies only CPI, GDP, and FR007 paths via the policy
# transmission map Θ_ν. Other variables (M2, NEER, US IP) are left at their
# baseline projections — this is a feature: the counterfactual operates only
# through the identified monetary policy channel.
#
# Implements both:
#   (A) Historical scenario (single projection from t*, à la Wolf Fig 9/10)
#   (B) Historical evolution (rolling forecast revisions, à la Wolf hist_evol)
#
# =============================================================================

using LinearAlgebra, Statistics, Printf, Dates, Serialization, Plots

include(joinpath(@__DIR__, "common.jl"))
const MAIN_DIR  = main_results_dir()
const INTER_DIR = intermediate_dir()

# =============================================================================
# 1) LOAD SAVED BVAR RESULTS (from LTP.jl)
# =============================================================================

# --- Load BVAR results ---
# Expected keys: A_list, c, Sigma_u, Psi, residuals, variable_names, dates, p, H
bvar = deserialize(joinpath(INTER_DIR, "bvar_results.jls"))

A_list     = bvar["A_list"]         # Vector of n×n lag matrices
c_vec      = bvar["c"]             # n-vector intercept
Sigma_u    = bvar["Sigma_u"]       # n×n residual covariance
Psi        = bvar["Psi"]           # Vector of n×n Wold MA matrices [Ψ₀, Ψ₁, ..., Ψ_H]
residuals  = bvar["residuals"]     # T_eff × n reduced-form residuals
var_labels = bvar["variable_names"]
dates_est  = bvar["dates"]
p_lag      = bvar["p"]
H_wold     = bvar["H"]

n = size(A_list[1], 1)             # number of variables (7)

println("="^70)
println("COUNTERFACTUAL ANALYSIS: STRICT CPI TARGETING (2%)")
println("="^70)
println("BVAR: $n variables, $p_lag lags, Wold horizon $H_wold")
println("Variables: ", join(var_labels, ", "))
println("Estimation sample: $(dates_est[1]) — $(dates_est[end])")

# =============================================================================
# 2) VARIABLE INDICES
# =============================================================================
# The BVAR ordering from LTP.jl is:
#   1: Real GDP YoY    2: IP YoY    3: CPI YoY    4: FR007
#   5: M2 YoY          6: NEER YoY  7: US IP YoY

gdp_idx    = 1   # Real GDP growth
ip_idx     = 2   # Industrial production growth
cpi_idx    = 3   # CPI (the targeting variable)
fr007_idx  = 4   # Policy rate (the instrument)
m2_idx     = 5
neer_idx   = 6
usip_idx   = 7

# IRF variable ordering (from RRShocks_monthly.jl / HFIShocks.jl):
#   1: realgdp_monthly_yoy  2: cpi  3: FR007  4: neer_yoy  5: IP_yoy
irf_gdp_idx   = 1
irf_cpi_idx   = 2
irf_fr007_idx = 3
irf_neer_idx  = 4
irf_ip_idx    = 5

println("\nTarget variable: $(var_labels[cpi_idx]) (index $cpi_idx)")
println("Policy instrument: $(var_labels[fr007_idx]) (index $fr007_idx)")

# =============================================================================
# 3) LOAD MONETARY POLICY IRFs — TWO SHOCK SERIES
# =============================================================================
#
# Following Wolf et al. (2025, Appendix D.2 / Figure D.2):
# The "empirics only" counterfactual uses TWO empirically estimated monetary
# policy shock series, each providing a different interest rate "treatment."
#
# In Wolf's US application:
#   Shock 1 = Romer & Romer (2004)     — short-lived rate path
#   Shock 2 = Aruoba & Drechsel (2024) — more persistent rate path
#
# In our China application:
#   Shock 1 = Narrative (Chen/Xiao/Zha policy rule residuals) — shorter-lived
#   Shock 2 = HFI (daily FR007 surprises on rate-change dates) — more persistent
#
# Together they span a 2-dimensional subspace of the policy causal effect space
# Θ_ν. The counterfactual solves for ν̃ ∈ ℝ^{2×T}: the optimal mix of the two
# treatments at each date.
#
# Convention: both IRFs normalized to +1pp FR007 on impact (contractionary).
# Shape: irf_draws is n_draws × (H+1) × n for each shock series.
#
# Save blocks needed in source scripts:
#   RRShocks_monthly.jl:
#     serialize("narrative_irf_results.jls", Dict(
#         "irf_point" => irf, "irf_draws" => irf_draws, "H" => H))
#   HFIShocks.jl:
#     serialize("hfi_irf_ratechange.jls", Dict(
#         "irf_point" => res_policy_change.irf,
#         "irf_draws" => irf_draws_from_res, "H" => H))

# --- Shock 1: Narrative (Chen/Xiao/Zha) ---
narr_data   = deserialize(joinpath(INTER_DIR, "narrative_irf_results.jls"))
narr_point  = narr_data["irf_point"]     # (H_narr+1) × n
narr_draws  = narr_data["irf_draws"]     # n_draws_narr × (H_narr+1) × n
H_narr      = narr_data["H"]

# --- Shock 2: HFI (rate-change-only) ---
hfi_data    = deserialize(joinpath(INTER_DIR, "hfi_irf_ratechange.jls"))
hfi_point   = hfi_data["irf_point"]      # (H_hfi+1) × n
hfi_draws   = hfi_data["irf_draws"]      # n_draws_hfi × (H_hfi+1) × n
H_hfi       = hfi_data["H"]

# Common IRF horizon: use the shorter of the two
H_irf = min(H_narr, H_hfi)

# Trim to common horizon
narr_point = narr_point[1:H_irf+1, :]
hfi_point  = hfi_point[1:H_irf+1, :]
narr_draws = narr_draws[:, 1:H_irf+1, :]
hfi_draws  = hfi_draws[:, 1:H_irf+1, :]

# Number of posterior draws: use the minimum across the two
n_draws_narr = size(narr_draws, 1)
n_draws_hfi  = size(hfi_draws, 1)
n_draws_irf  = min(n_draws_narr, n_draws_hfi)

n_shocks = 2   # two empirical shock series

println("\nShock 1 (Narrative): horizon $H_narr, $n_draws_narr posterior draws")
println("Shock 2 (HFI):      horizon $H_hfi, $n_draws_hfi posterior draws")
println("Common horizon: $H_irf, using $n_draws_irf paired draws")

# =============================================================================
# 4) DATA: FULL SAMPLE FOR HISTORY + FORECASTING
# =============================================================================
# LTP.jl saves dates and Y_full in bvar_results.jls, but the estimation sample
# may end before the counterfactual window. We need the raw data to get
# actual realizations during the counterfactual period.
# NOTE: you must also serialize dates_full and Y_full from LTP.jl:
#   results["dates_full"] = dates_full
#   results["Y_full"]     = Y_full

# Try loading from bvar_results first; fall back to CSV if not present
if haskey(bvar, "dates_full") && haskey(bvar, "Y_full")
    dates_all = bvar["dates_full"]
    Y_all     = bvar["Y_full"]
else
    # Fallback: reload from CSV
    using CSV, DataFrames
    _df = CSV.read(joinpath(DERIVED_DIR, "china_longterm_data.csv"), DataFrame)
    _df.date = Date.(_df.date); sort!(_df, :date)
    _syms = [:realgdp_monthly_yoy, :IP_yoy, :cpi, :FR007, :M2_growth, :neer_yoy, :US_IP_yoy]
    _df = dropmissing(_df, _syms)
    dates_all = _df.date
    Y_all     = Matrix{Float64}(_df[:, _syms])
end

# =============================================================================
# 5) COUNTERFACTUAL RULE DEFINITIONS
# =============================================================================
#
# Each rule is a NamedTuple with:
#   label  — human-readable name (for plot titles / filenames)
#   gap_fn — function(baseline_fc, fcst_hor) → T-vector "gap" that the
#            transmission map must close. gap = target_path - baseline_path.
#
# To add a new rule, just add an entry to `rules` below.

# --- Rule library ---

function rule_strict_cpi_targeting(; target=2.0)
    (
        label  = "Strict CPI Targeting ($(target)%)",
        gap_fn = (bl, T) -> fill(target, T) .- bl[:, cpi_idx],
    )
end

# Time-varying strict targeting: actual government work report targets
# CPI: 3% (2021–2024), 2% (2025)  |  GDP: 5% (all years, ~6% for 2021)
function rule_strict_govt_targets(; cnfctl_dates_ref=nothing)
    (
        label  = "Strict Gov't Targets (CPI 3/3/3/3/2%)",
        gap_fn = (bl, T) -> begin
            pi_target = zeros(T)
            for t in 1:T
                yr = year(cnfctl_dates_ref[t])
                pi_target[t] = yr <= 2024 ? 3.0 : 2.0
            end
            return pi_target .- bl[:, cpi_idx]
        end,
    )
end

# =============================================================================
# FLEXIBLE LOSS FUNCTION COUNTERFACTUAL
# =============================================================================
#
# Minimize:
#   L = λ_π ‖π_base + Π_m ν − π*‖² + λ_y ‖y_base + Y_m ν − y*‖²
#     + λ_i ‖D(i_base + I_m ν)‖²   + λ_e ‖E_m ν‖²
#
# where D is the first-difference matrix (rate smoothing),
# and E_m ν penalizes monetary-policy-induced NEER YoY deviations from baseline.
#
# This is a standard weighted least squares problem in ν — closed-form solution.

function build_first_diff_matrix(T_hor::Int)
    D = zeros(T_hor - 1, T_hor)
    for t in 1:T_hor-1
        D[t, t]   = -1.0
        D[t, t+1] =  1.0
    end
    return D
end

function cnfctl_flexible(pi_base, y_base, i_base,
                         Pi_m, Y_m, I_m, E_m,
                         pi_target, y_target,
                         λ_pi, λ_y, λ_i, λ_e)
    T = length(pi_base)

    gap_pi = pi_target .- pi_base
    gap_y  = y_target  .- y_base

    D       = build_first_diff_matrix(T)
    DI      = D * I_m
    Di_base = D * i_base

    sq_pi = sqrt(λ_pi); sq_y = sqrt(λ_y); sq_i = sqrt(λ_i); sq_e = sqrt(λ_e)

    A_stack = vcat(sq_pi * Pi_m, sq_y * Y_m, sq_i * DI, sq_e * E_m)
    b_stack = vcat(sq_pi * gap_pi, sq_y * gap_y, sq_i * (-Di_base), sq_e * zeros(T))

    nu_tilde  = A_stack \ b_stack
    pi_cnfctl = pi_base .+ Pi_m * nu_tilde
    y_cnfctl  = y_base  .+ Y_m  * nu_tilde
    i_cnfctl  = i_base  .+ I_m  * nu_tilde
    e_cnfctl  = E_m * nu_tilde

    L_pi = λ_pi * sum((pi_cnfctl .- pi_target).^2)
    L_y  = λ_y  * sum((y_cnfctl  .- y_target).^2)
    L_i  = λ_i  * sum((D * i_cnfctl).^2)
    L_e  = λ_e  * sum(e_cnfctl.^2)

    return (pi_cnfctl=pi_cnfctl, y_cnfctl=y_cnfctl, i_cnfctl=i_cnfctl,
            e_cnfctl=e_cnfctl, nu_tilde=nu_tilde,
            loss=Dict("total"=>L_pi+L_y+L_i+L_e, "pi"=>L_pi, "y"=>L_y, "i"=>L_i, "e"=>L_e))
end

# Posterior bands for the flexible counterfactual
function posterior_bands_flex(pi_base, y_base, i_base,
                              Pi_m_pt, Y_m_pt, I_m_pt, E_m_pt,
                              narr_draws, hfi_draws, n_draws_irf, fcst_hor,
                              pi_target, y_target, λ_pi, λ_y, λ_i, λ_e,
                              cpi_idx, gdp_idx, fr007_idx, neer_idx)
    pi_d = zeros(fcst_hor, n_draws_irf)
    y_d  = zeros(fcst_hor, n_draws_irf)
    i_d  = zeros(fcst_hor, n_draws_irf)
    e_d  = zeros(fcst_hor, n_draws_irf)
    nv = 0
    for d in 1:n_draws_irf
        Pi_d = build_two_shock_map(narr_draws[d,:,irf_cpi_idx],   hfi_draws[d,:,irf_cpi_idx],   fcst_hor)
        Y_d  = build_two_shock_map(narr_draws[d,:,irf_gdp_idx],   hfi_draws[d,:,irf_gdp_idx],   fcst_hor)
        I_d  = build_two_shock_map(narr_draws[d,:,irf_fr007_idx], hfi_draws[d,:,irf_fr007_idx], fcst_hor)
        E_d  = build_two_shock_map(narr_draws[d,:,irf_neer_idx],  hfi_draws[d,:,irf_neer_idx],  fcst_hor)
        try
            res = cnfctl_flexible(pi_base, y_base, i_base, Pi_d, Y_d, I_d, E_d,
                                  pi_target, y_target, λ_pi, λ_y, λ_i, λ_e)
            pi_d[:,d] = res.pi_cnfctl; y_d[:,d] = res.y_cnfctl
            i_d[:,d]  = res.i_cnfctl;  e_d[:,d] = res.e_cnfctl
            nv += 1
        catch
            pi_d[:,d] .= NaN; y_d[:,d] .= NaN; i_d[:,d] .= NaN; e_d[:,d] .= NaN
        end
    end
    println("  Valid draws: $nv / $n_draws_irf")
    return (;
        pi_lb = nanquantile(pi_d, 0.16), pi_ub = nanquantile(pi_d, 0.84),
        y_lb  = nanquantile(y_d,  0.16), y_ub  = nanquantile(y_d,  0.84),
        i_lb  = nanquantile(i_d,  0.16), i_ub  = nanquantile(i_d,  0.84),
        e_lb  = nanquantile(e_d,  0.16), e_ub  = nanquantile(e_d,  0.84),
    )
end

# --- Rules are defined after cnfctl_dates (see run sections below) ---

# =============================================================================
# 6) COUNTERFACTUAL SETTINGS & BASELINE — SHARED INFRASTRUCTURE
# =============================================================================

function forecast_from_date(Y, A_list, c, start_idx, H_fc)
    p_loc = length(A_list); n_loc = size(Y, 2)
    history = [Y[start_idx - j + 1, :] for j in 1:p_loc]
    fc = zeros(H_fc, n_loc)
    for h in 1:H_fc
        y_hat = copy(c)
        for j in 1:min(p_loc, length(history)); y_hat .+= A_list[j] * history[j]; end
        fc[h, :] = y_hat
        pushfirst!(history, y_hat); pop!(history)
    end
    return fc
end

# CPI target by year (government work report)
function cpi_target_for_year(yr::Int)
    yr <= 2024 ? 3.0 : 2.0
end

# GDP target by year
function gdp_target_for_year(yr::Int)
    yr == 2021 ? 6.0 : 5.0   # 2021 was ">6%", rest "~5%"
end

# =============================================================================
# 6) BUILD POLICY TRANSMISSION MAP Θ_ν — TWO SHOCK SERIES
# =============================================================================
#
# Following Wolf et al.: with n_ν = 2 empirical shock series, the transmission
# map for each variable is T × 2S, formed by horizontally concatenating the
# Toeplitz maps from each shock series:
#
#   Pi_m = [ Pi_m_narr  Pi_m_hfi ]    — T × 2S
#
# where Pi_m_narr[h, s] = narrative CPI IRF at lag (h-s), and similarly for HFI.
# The shock vector ν̃ is then 2S × 1, with the first S entries being the
# narrative treatment intensities and the next S entries the HFI treatment
# intensities at each date.
#
# This gives the optimizer a 2-dimensional policy space at each date: it can
# mix a "narrative-type" rate path (shorter-lived) with an "HFI-type" rate
# path (more persistent) to best achieve the CPI target.

function build_transmission_map(irf_col::Vector{Float64}, T_hor::Int)
    S = min(T_hor, length(irf_col))  # shock space dimension per series
    M = zeros(T_hor, S)
    for s in 1:S
        for h in s:T_hor
            lag = h - s + 1
            if lag <= length(irf_col)
                M[h, s] = irf_col[lag]
            end
        end
    end
    return M
end

function build_two_shock_map(irf_col_1::Vector{Float64}, irf_col_2::Vector{Float64},
                             T_hor::Int)
    M1 = build_transmission_map(irf_col_1, T_hor)
    M2 = build_transmission_map(irf_col_2, T_hor)
    return hcat(M1, M2)   # T × 2S
end

# Transmission maps are built per-scenario (fcst_hor varies)
# Functions defined above: build_transmission_map, build_two_shock_map

# =============================================================================
# 8) COUNTERFACTUAL SOLVER
# =============================================================================

function cnfctl_scenario(gap::Vector{Float64}, y_base::Vector{Float64},
                         i_base::Vector{Float64}, pi_base::Vector{Float64},
                         Pi_m::Matrix{Float64}, Y_m::Matrix{Float64},
                         I_m::Matrix{Float64})
    nu_tilde  = Pi_m \ gap
    pi_cnfctl = pi_base .+ Pi_m * nu_tilde
    y_cnfctl  = y_base  .+ Y_m  * nu_tilde
    i_cnfctl  = i_base  .+ I_m  * nu_tilde
    return pi_cnfctl, y_cnfctl, i_cnfctl, nu_tilde
end

# =============================================================================
# 9) POSTERIOR BANDS — HELPER
# =============================================================================

function nanquantile(X::Matrix{Float64}, q::Float64)
    [let v = filter(!isnan, X[t,:]); isempty(v) ? NaN : quantile(v, q) end
     for t in 1:size(X,1)]
end

function posterior_bands(gap, pi_base, y_base, i_base,
                         narr_draws, hfi_draws, n_draws_irf, fcst_hor,
                         cpi_idx, gdp_idx, fr007_idx)
    pi_d = zeros(fcst_hor, n_draws_irf)
    y_d  = zeros(fcst_hor, n_draws_irf)
    i_d  = zeros(fcst_hor, n_draws_irf)
    nv = 0
    for d in 1:n_draws_irf
        Pi_d = build_two_shock_map(narr_draws[d,:,irf_cpi_idx],   hfi_draws[d,:,irf_cpi_idx],   fcst_hor)
        Y_d  = build_two_shock_map(narr_draws[d,:,irf_gdp_idx],   hfi_draws[d,:,irf_gdp_idx],   fcst_hor)
        I_d  = build_two_shock_map(narr_draws[d,:,irf_fr007_idx], hfi_draws[d,:,irf_fr007_idx], fcst_hor)
        try
            p, y, i, _ = cnfctl_scenario(gap, y_base, i_base, pi_base, Pi_d, Y_d, I_d)
            pi_d[:,d] = p; y_d[:,d] = y; i_d[:,d] = i; nv += 1
        catch; pi_d[:,d] .= NaN; y_d[:,d] .= NaN; i_d[:,d] .= NaN; end
    end
    println("  Valid draws: $nv / $n_draws_irf")
    bands = (;
        pi_med = nanquantile(pi_d, 0.5), pi_lb = nanquantile(pi_d, 0.16), pi_ub = nanquantile(pi_d, 0.84),
        y_med  = nanquantile(y_d, 0.5),  y_lb  = nanquantile(y_d, 0.16),  y_ub  = nanquantile(y_d, 0.84),
        i_med  = nanquantile(i_d, 0.5),  i_lb  = nanquantile(i_d, 0.16),  i_ub  = nanquantile(i_d, 0.84),
    )
    return bands
end

# =============================================================================
# 10) D.2-STYLE PLOT FUNCTION
# =============================================================================

function plot_counterfactual(rule_label, cnfctl_dates, pi_cnfctl, y_cnfctl, i_cnfctl,
                             bands, pi_base, y_base, i_base,
                             hist_dates, hist_data, actual_dates, actual_data,
                             cpi_idx, gdp_idx, fr007_idx;
                             save_prefix="counterfactual")

    blue = RGB(0.45, 0.62, 0.70)
    lblue = RGB(0.45, 0.62, 0.70)

    fig = plot(layout=(1, 3), size=(1400, 450), margin=8Plots.mm,
        plot_title=rule_label * " — from $(cnfctl_dates[1])")

    # Helper: one panel
    function _panel!(sp, var_idx, pt_cnfctl, base, bd_lb, bd_ub, title_str)
        plot!(fig[sp], hist_dates, hist_data[:, var_idx],
            color=:black, lw=2.5, label="Data")
        plot!(fig[sp], actual_dates, actual_data[:, var_idx],
            color=:black, lw=2.5, label="")
        plot!(fig[sp], cnfctl_dates, base,
            color=:gray, lw=2, ls=:dash, label="Forecast")
        plot!(fig[sp], cnfctl_dates, bd_lb,
            fillrange=bd_ub, fillalpha=0.2, fillcolor=lblue, lw=0, label="68%")
        plot!(fig[sp], cnfctl_dates, pt_cnfctl,
            color=blue, lw=2.5, label="Counterfact'l")
        title!(fig[sp], title_str)
    end

    _panel!(1, cpi_idx,   pi_cnfctl, pi_base, bands.pi_lb, bands.pi_ub, "CPI YoY (%)")
    _panel!(2, gdp_idx,   y_cnfctl,  y_base,  bands.y_lb,  bands.y_ub,  "Real GDP YoY (%)")
    _panel!(3, fr007_idx, i_cnfctl,  i_base,  bands.i_lb,  bands.i_ub,  "FR007 (%)")

    display(fig)
    fname = replace(lowercase(save_prefix), " " => "_") * ".png"
    savefig(fig, joinpath(MAIN_DIR, fname))
    println("  Saved: $fname")
    return fig
end

# =============================================================================
# 11) RUN COUNTERFACTUAL SCENARIOS
# =============================================================================
#
# Two counterfactual periods:
#   (A) 2023:01–2025:12  — the deflation episode
#   (B) 2021:01–2025:12  — full post-COVID period (includes recovery + deflation)
#
# Each runs: (1) strict CPI targeting, (2) flexible IT (balanced)

scenarios = [
    (start=Date(2023,1,1), stop=Date(2025,12,1), n_hist=12, label="2023"),
    (start=Date(2021,1,1), stop=Date(2025,12,1), n_hist=12, label="2021"),
]

# Balanced flexible IT lambdas
λ_pi = 1.0; λ_y = 0.5; λ_i = 1.0; λ_e = 0.5

all_scenario_results = Dict{String, Any}()

for sc in scenarios
    println("\n" * "#"^70)
    println("# SCENARIO: counterfactual from $(sc.start)")
    println("#"^70)

    # --- Setup for this scenario ---
    cnfctl_start = sc.start
    cnfctl_end   = sc.stop
    fcst_date_idx = findfirst(d -> d >= cnfctl_start, dates_all)
    fcst_hor      = length(cnfctl_start:Month(1):cnfctl_end)
    cnfctl_dates  = [cnfctl_start + Month(h-1) for h in 1:fcst_hor]

    baseline_fc = forecast_from_date(Y_all, A_list, c_vec, fcst_date_idx - 1, fcst_hor)

    T_actual     = min(fcst_hor, size(Y_all, 1) - fcst_date_idx + 1)
    actual_data  = Y_all[fcst_date_idx:min(fcst_date_idx + T_actual - 1, end), :]
    actual_dates = dates_all[fcst_date_idx:min(fcst_date_idx + T_actual - 1, end)]

    hist_start_idx = max(1, fcst_date_idx - sc.n_hist)
    hist_dates     = dates_all[hist_start_idx:fcst_date_idx-1]
    hist_data      = Y_all[hist_start_idx:fcst_date_idx-1, :]

    # Build transmission maps for this horizon
    Pi_m = build_two_shock_map(narr_point[:, irf_cpi_idx],   hfi_point[:, irf_cpi_idx],   fcst_hor)
    Y_m  = build_two_shock_map(narr_point[:, irf_gdp_idx],   hfi_point[:, irf_gdp_idx],   fcst_hor)
    I_m  = build_two_shock_map(narr_point[:, irf_fr007_idx], hfi_point[:, irf_fr007_idx], fcst_hor)
    E_m  = build_two_shock_map(narr_point[:, irf_neer_idx],  hfi_point[:, irf_neer_idx],  fcst_hor)
    S_per = size(Pi_m, 2) ÷ 2

    pi_base = baseline_fc[:, cpi_idx]
    y_base  = baseline_fc[:, gdp_idx]
    i_base  = baseline_fc[:, fr007_idx]
    e_base  = baseline_fc[:, neer_idx]

    println("  Horizon: $fcst_hor months, maps: $(size(Pi_m))")

    # --- Time-varying targets ---
    pi_target = [cpi_target_for_year(year(d)) for d in cnfctl_dates]
    y_target  = [gdp_target_for_year(year(d)) for d in cnfctl_dates]

    # =====================================================================
    # (A) STRICT CPI TARGETING (for comparison — expected to blow up)
    # =====================================================================
    println("\n--- Strict CPI targeting ---")
    gap = pi_target .- pi_base
    pi_c, y_c, i_c, nu = cnfctl_scenario(gap, y_base, i_base, pi_base, Pi_m, Y_m, I_m)
    nu_narr = nu[1:S_per]; nu_hfi = nu[S_per+1:end]
    println(@sprintf("  Narrative: cumul %+.0f bps  |  HFI: cumul %+.0f bps",
        sum(nu_narr)*100, sum(nu_hfi)*100))
    println(@sprintf("  FR007 range: [%.1f, %.1f]", minimum(i_c), maximum(i_c)))

    # Posterior bands (strict)
    bands_strict = posterior_bands(gap, pi_base, y_base, i_base,
                narr_draws, hfi_draws, n_draws_irf, fcst_hor,
                cpi_idx, gdp_idx, fr007_idx)

    # Plot strict (3-panel, Wolf D.2 style)
    plot_counterfactual("Strict Gov't CPI Targets — from $(sc.start)", cnfctl_dates,
        pi_c, y_c, i_c, bands_strict, pi_base, y_base, i_base,
        hist_dates, hist_data, actual_dates, actual_data,
        cpi_idx, gdp_idx, fr007_idx;
        save_prefix="cnfctl_strict_$(sc.label)")

    # =====================================================================
    # (B) FLEXIBLE IT — BALANCED (λ_π=1, λ_y=0.5, λ_i=1, λ_e=0.5)
    # =====================================================================
    println("\n--- Flexible IT (balanced) ---")
    res = cnfctl_flexible(pi_base, y_base, i_base, Pi_m, Y_m, I_m, E_m,
                          pi_target, y_target, λ_pi, λ_y, λ_i, λ_e)

    println(@sprintf("  Loss: total=%.1f  π=%.1f  y=%.1f  Δi=%.1f  e=%.1f",
        res.loss["total"], res.loss["pi"], res.loss["y"], res.loss["i"], res.loss["e"]))
    println(@sprintf("  CPI:   mean=%.2f%%  range=[%.2f, %.2f]",
        mean(res.pi_cnfctl), minimum(res.pi_cnfctl), maximum(res.pi_cnfctl)))
    println(@sprintf("  FR007: mean=%.2f%%  range=[%.2f, %.2f]",
        mean(res.i_cnfctl), minimum(res.i_cnfctl), maximum(res.i_cnfctl)))

    # Posterior bands (flexible)
    println("  Computing posterior bands...")
    bands_flex = posterior_bands_flex(pi_base, y_base, i_base,
                    Pi_m, Y_m, I_m, E_m,
                    narr_draws, hfi_draws, n_draws_irf, fcst_hor,
                    pi_target, y_target, λ_pi, λ_y, λ_i, λ_e,
                    irf_cpi_idx, irf_gdp_idx, irf_fr007_idx, irf_neer_idx)

    # 4-panel plot with 68% bands
    blue  = RGB(0.45, 0.62, 0.70)
    lblue = RGB(0.45, 0.62, 0.70)

    fig = plot(layout=(2, 2), size=(1200, 700), margin=8Plots.mm,
        plot_title="Flexible IT (Balanced) — from $(sc.start)")

    # Helper for flex panels
    function _flex_panel!(sp, hist_var_idx, pt_cnfctl, base, bd_lb, bd_ub,
                          title_str; target_line=nothing, cnfctl_is_deviation=false)
        plot!(fig[sp], hist_dates, hist_data[:, hist_var_idx],
            color=:black, lw=2.5, label="Data")
        plot!(fig[sp], actual_dates, actual_data[:, hist_var_idx],
            color=:black, lw=2.5, label="")
        plot!(fig[sp], cnfctl_dates, base,
            color=:gray, lw=2, ls=:dash, label="Forecast")
        if target_line !== nothing
            plot!(fig[sp], cnfctl_dates, target_line,
                color=:red, lw=1.5, ls=:dot, label="Target")
        end
        plot!(fig[sp], cnfctl_dates, bd_lb,
            fillrange=bd_ub, fillalpha=0.2, fillcolor=lblue, lw=0, label="68%")
        plot!(fig[sp], cnfctl_dates, pt_cnfctl,
            color=blue, lw=2.5, label="Counterfact'l")
        title!(fig[sp], title_str)
    end

    _flex_panel!(1, cpi_idx,   res.pi_cnfctl, pi_base,
        bands_flex.pi_lb, bands_flex.pi_ub, "CPI YoY (%)"; target_line=pi_target)
    _flex_panel!(2, gdp_idx,   res.y_cnfctl,  y_base,
        bands_flex.y_lb,  bands_flex.y_ub,  "Real GDP YoY (%)"; target_line=y_target)
    _flex_panel!(3, fr007_idx, res.i_cnfctl,  i_base,
        bands_flex.i_lb,  bands_flex.i_ub,  "FR007 (%)")
    _flex_panel!(4, neer_idx,  e_base .+ res.e_cnfctl, e_base,
        e_base .+ bands_flex.e_lb, e_base .+ bands_flex.e_ub, "NEER YoY (%)")

    display(fig)
    fname = "cnfctl_flex_balanced_$(sc.label).png"
    savefig(fig, joinpath(MAIN_DIR, fname))
    println("  Saved: $fname")

    # Store results
    all_scenario_results[sc.label] = Dict(
        "cnfctl_start" => cnfctl_start, "cnfctl_dates" => cnfctl_dates,
        "baseline_fc" => baseline_fc,
        "actual_data" => actual_data, "actual_dates" => actual_dates,
        "hist_data" => hist_data, "hist_dates" => hist_dates,
        "pi_target" => pi_target, "y_target" => y_target,
        # Strict
        "strict" => Dict("pi"=>pi_c, "y"=>y_c, "i"=>i_c, "nu"=>nu, "bands"=>bands_strict),
        # Flexible
        "flex" => Dict("res"=>res, "bands"=>bands_flex),
        # Maps
        "Pi_m"=>Pi_m, "Y_m"=>Y_m, "I_m"=>I_m, "E_m"=>E_m, "S_per"=>S_per,
    )
end

# =============================================================================
# 12) SAVE
# =============================================================================

serialize(joinpath(INTER_DIR, "counterfactual_results.jls"), Dict(
    "scenarios"     => all_scenario_results,
    "lambda_config" => Dict("λ_pi"=>λ_pi, "λ_y"=>λ_y, "λ_i"=>λ_i, "λ_e"=>λ_e),
    "n_shocks"      => n_shocks,
))
println("\nResults saved to counterfactual_results.jls")
println("✓ Counterfactual analysis complete.")
