##
# =============================================================================
# COUNTERFACTUAL HISTORICAL SCENARIOS — LOSS FUNCTION APPROACH
# =============================================================================
#
# Following Wolf et al. (2025) — loss function counterfactual
#
# Method: minimize weighted loss L with FR007 smoothing, solved in closed form.
#   L = λ_π ‖π - π*‖² + λ_y ‖y - y*‖² + λ_i ‖Δi‖² + λ_e ‖E_m ν‖²
#
# Three targeting rules (all include FR007 smoothing):
#   1. CPI only:          λ_π=1, λ_y=0, λ_i=1, λ_e=0
#   2. CPI + GDP:         λ_π=1, λ_y=0.5, λ_i=1, λ_e=0
#   3. CPI + GDP + NEER:  λ_π=1, λ_y=0.5, λ_i=1, λ_e=0.5
#
# Three year configurations (counterfactual start → estimation sample):
#   cfctl 2020 → sample 2019   (pre-COVID IRFs, counterfactual through COVID)
#   cfctl 2023 → sample 2022   (pre-deflation IRFs)
#   cfctl 2023 → sample 2025   (full-sample IRFs)
#
# =============================================================================

using LinearAlgebra, Statistics, Printf, Dates, Serialization, Plots
using CSV, DataFrames

include(joinpath(@__DIR__, "common.jl"))

# =============================================================================
# 1) VARIABLE INDEX MAPPINGS
# =============================================================================

# BVAR ordering (LTP.jl): 7 variables
#   1: Real GDP YoY   2: IP YoY   3: CPI YoY   4: FR007
#   5: M2 YoY         6: NEER YoY 7: US IP YoY
const gdp_idx   = 1
const cpi_idx   = 3
const fr007_idx = 4
const neer_idx  = 6

# IRF ordering (RRShocks/HFIShocks): 5 variables
#   1: GDP   2: CPI   3: FR007   4: NEER   5: IP
const irf_gdp_idx   = 1
const irf_cpi_idx   = 2
const irf_fr007_idx = 3
const irf_neer_idx  = 4

# =============================================================================
# 2) LOAD ACTUAL DATA (full sample, for overlay on all plots)
# =============================================================================

_df = CSV.read(joinpath(DERIVED_DIR, "china_longterm_data.csv"), DataFrame)
_df.date = Date.(_df.date); sort!(_df, :date)
_syms = [:realgdp_monthly_yoy, :IP_yoy, :cpi, :FR007, :M2_growth, :neer_yoy, :US_IP_yoy]
_df = dropmissing(_df, _syms)
const dates_actual = _df.date
const Y_actual     = Matrix{Float64}(_df[:, _syms])

# =============================================================================
# 3) SHARED FUNCTIONS
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

function cpi_target_for_year(yr::Int)
    yr <= 2024 ? 3.0 : 2.0
end

function gdp_target_for_year(yr::Int)
    yr == 2021 ? 6.0 : 5.0
end

function build_transmission_map(irf_col::Vector{Float64}, T_hor::Int)
    S = min(T_hor, length(irf_col))
    M = zeros(T_hor, S)
    for s in 1:S, h in s:T_hor
        lag = h - s + 1
        if lag <= length(irf_col)
            M[h, s] = irf_col[lag]
        end
    end
    return M
end

function build_two_shock_map(irf_col_1::Vector{Float64}, irf_col_2::Vector{Float64},
                             T_hor::Int)
    hcat(build_transmission_map(irf_col_1, T_hor),
         build_transmission_map(irf_col_2, T_hor))
end

function build_first_diff_matrix(T_hor::Int)
    D = zeros(T_hor - 1, T_hor)
    for t in 1:T_hor-1
        D[t, t] = -1.0; D[t, t+1] = 1.0
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

function nanquantile(X::Matrix{Float64}, q::Float64)
    [let v = filter(!isnan, X[t,:]); isempty(v) ? NaN : quantile(v, q) end
     for t in 1:size(X,1)]
end

function posterior_bands_flex(pi_base, y_base, i_base,
                              narr_draws, hfi_draws, n_draws_irf, fcst_hor,
                              pi_target, y_target, λ_pi, λ_y, λ_i, λ_e)
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

# =============================================================================
# 4) CONFIGURATION
# =============================================================================

# Year configurations: (counterfactual start, end, estimation sample label)
year_configs = [
    (cfctl_start=Date(2020,1,1), cfctl_end=Date(2025,12,1), sample="2019", n_hist=12),
    (cfctl_start=Date(2023,1,1), cfctl_end=Date(2025,12,1), sample="2022", n_hist=12),
    (cfctl_start=Date(2023,1,1), cfctl_end=Date(2025,12,1), sample="2025", n_hist=12),
]

# Targeting rules (rows in each figure): all include FR007 smoothing
rules = [
    (label="CPI Targeting",       λ_π=1.0, λ_y=0.0, λ_i=1.0, λ_e=0.0),
    (label="CPI + GDP Targeting", λ_π=1.0, λ_y=0.5, λ_i=1.0, λ_e=0.0),
    (label="CPI + GDP + NEER",    λ_π=1.0, λ_y=0.5, λ_i=1.0, λ_e=0.5),
]

# =============================================================================
# 5) MAIN LOOP — one combined 3×3 figure per year config
# =============================================================================

all_results = Dict{String, Any}()

for yc in year_configs

    println("\n", "="^70)
    println("  Loading estimation results: sample=$(yc.sample)")
    println("  Counterfactual: $(yc.cfctl_start) → $(yc.cfctl_end)")
    println("="^70)

    INTER_DIR = intermediate_dir(yc.sample)
    MAIN_DIR  = main_results_dir(yc.sample)

    # --- Load BVAR baseline ---
    bvar = deserialize(joinpath(INTER_DIR, "bvar_results.jls"))
    A_list     = bvar["A_list"]
    c_vec      = bvar["c"]
    var_labels = bvar["variable_names"]
    n = size(A_list[1], 1)

    println("BVAR: $n variables, $(bvar["p"]) lags")

    # --- Load IRFs ---
    narr_data  = deserialize(joinpath(INTER_DIR, "narrative_irf_results.jls"))
    hfi_data   = deserialize(joinpath(INTER_DIR, "hfi_irf_ratechange.jls"))

    narr_point = narr_data["irf_point"]
    narr_draws = narr_data["irf_draws"]
    hfi_point  = hfi_data["irf_point"]
    hfi_draws  = hfi_data["irf_draws"]

    H_irf = min(narr_data["H"], hfi_data["H"])
    narr_point = narr_point[1:H_irf+1, :]
    hfi_point  = hfi_point[1:H_irf+1, :]
    narr_draws = narr_draws[:, 1:H_irf+1, :]
    hfi_draws  = hfi_draws[:, 1:H_irf+1, :]
    n_draws_irf = min(size(narr_draws, 1), size(hfi_draws, 1))

    println("IRFs: H=$H_irf, $n_draws_irf paired draws")

    # --- Baseline forecast setup ---
    fcst_date_idx = findfirst(d -> d >= yc.cfctl_start, dates_actual)
    fcst_hor      = length(yc.cfctl_start:Month(1):yc.cfctl_end)
    cnfctl_dates  = [yc.cfctl_start + Month(h-1) for h in 1:fcst_hor]

    baseline_fc = forecast_from_date(Y_actual, A_list, c_vec, fcst_date_idx - 1, fcst_hor)

    T_actual_avail = min(fcst_hor, size(Y_actual, 1) - fcst_date_idx + 1)
    actual_data  = Y_actual[fcst_date_idx:min(fcst_date_idx + T_actual_avail - 1, end), :]
    actual_dates_window = dates_actual[fcst_date_idx:min(fcst_date_idx + T_actual_avail - 1, end)]

    hist_start_idx = max(1, fcst_date_idx - yc.n_hist)
    hist_dates = dates_actual[hist_start_idx:fcst_date_idx-1]
    hist_data  = Y_actual[hist_start_idx:fcst_date_idx-1, :]

    pi_base = baseline_fc[:, cpi_idx]
    y_base  = baseline_fc[:, gdp_idx]
    i_base  = baseline_fc[:, fr007_idx]
    e_base  = baseline_fc[:, neer_idx]

    pi_target = [cpi_target_for_year(year(d)) for d in cnfctl_dates]
    y_target  = [gdp_target_for_year(year(d)) for d in cnfctl_dates]

    Pi_m = build_two_shock_map(narr_point[:, irf_cpi_idx],   hfi_point[:, irf_cpi_idx],   fcst_hor)
    Y_m  = build_two_shock_map(narr_point[:, irf_gdp_idx],   hfi_point[:, irf_gdp_idx],   fcst_hor)
    I_m  = build_two_shock_map(narr_point[:, irf_fr007_idx], hfi_point[:, irf_fr007_idx], fcst_hor)
    E_m  = build_two_shock_map(narr_point[:, irf_neer_idx],  hfi_point[:, irf_neer_idx],  fcst_hor)

    println("  Horizon: $fcst_hor months, maps: $(size(Pi_m))")

    # --- Solve all 3 rules, collect results ---
    cfctl_year = string(year(yc.cfctl_start))
    rule_results = []  # Vector of (res, bands) tuples

    for rule in rules
        key = "$(yc.sample)_$(cfctl_year)_$(rule.label)"

        println("\n", "#"^60)
        println("# $(rule.label)  |  cfctl $(cfctl_year) → sample $(yc.sample)")
        println("#"^60)

        res = cnfctl_flexible(pi_base, y_base, i_base, Pi_m, Y_m, I_m, E_m,
                              pi_target, y_target,
                              rule.λ_π, rule.λ_y, rule.λ_i, rule.λ_e)

        println(@sprintf("  Loss: total=%.1f  π=%.1f  y=%.1f  Δi=%.1f  e=%.1f",
            res.loss["total"], res.loss["pi"], res.loss["y"], res.loss["i"], res.loss["e"]))
        println(@sprintf("  CPI:   mean=%.2f%%  range=[%.2f, %.2f]",
            mean(res.pi_cnfctl), minimum(res.pi_cnfctl), maximum(res.pi_cnfctl)))
        println(@sprintf("  FR007: mean=%.2f%%  range=[%.2f, %.2f]",
            mean(res.i_cnfctl), minimum(res.i_cnfctl), maximum(res.i_cnfctl)))

        println("  Computing posterior bands...")
        bands = posterior_bands_flex(pi_base, y_base, i_base,
                    narr_draws, hfi_draws, n_draws_irf, fcst_hor,
                    pi_target, y_target,
                    rule.λ_π, rule.λ_y, rule.λ_i, rule.λ_e)

        push!(rule_results, (res=res, bands=bands))

        all_results[key] = Dict(
            "cfctl_start" => yc.cfctl_start, "sample" => yc.sample,
            "rule" => rule.label, "lambdas" => (rule.λ_π, rule.λ_y, rule.λ_i, rule.λ_e),
            "cnfctl_dates" => cnfctl_dates,
            "res" => res, "bands" => bands,
            "baseline_fc" => baseline_fc,
            "pi_target" => pi_target, "y_target" => y_target,
        )
    end

    # --- Build combined 3×4 figure (rows=rules, cols=CPI/GDP/FR007/NEER) ---
    blue  = RGB(0.45, 0.62, 0.70)
    lblue = RGB(0.45, 0.62, 0.70)
    n_cols = 4

    fig = plot(layout=(3, n_cols), size=(1800, 1000), margin=6Plots.mm,
        plot_title="Counterfactual from $(cfctl_year) — sample $(yc.sample)")

    # Continuous actual data line: history + counterfactual window (no gap)
    all_plot_dates = vcat(hist_dates, actual_dates_window)
    all_plot_data  = vcat(hist_data, actual_data)

    # Columns: CPI, GDP, FR007, NEER
    col_vars = [
        (idx=cpi_idx,   cnfctl=:pi_cnfctl, base=pi_base, e_adj=false, lb=:pi_lb, ub=:pi_ub, tgt=pi_target,  col_title="CPI YoY (%)"),
        (idx=gdp_idx,   cnfctl=:y_cnfctl,  base=y_base,  e_adj=false, lb=:y_lb,  ub=:y_ub,  tgt=y_target,   col_title="Real GDP YoY (%)"),
        (idx=fr007_idx, cnfctl=:i_cnfctl,  base=i_base,  e_adj=false, lb=:i_lb,  ub=:i_ub,  tgt=nothing,    col_title="FR007 (%)"),
        (idx=neer_idx,  cnfctl=:e_cnfctl,  base=e_base,  e_adj=true,  lb=:e_lb,  ub=:e_ub,  tgt=nothing,    col_title="NEER YoY (%)"),
    ]

    for (r, (rule, rr)) in enumerate(zip(rules, rule_results))
        for (c, cv) in enumerate(col_vars)
            sp = (r - 1) * n_cols + c

            # Actual data — one continuous line
            plot!(fig[sp], all_plot_dates, all_plot_data[:, cv.idx],
                color=:black, lw=2, label=(r==1 && c==1 ? "Data" : ""))

            # Baseline forecast
            plot!(fig[sp], cnfctl_dates, cv.base,
                color=:gray, lw=1.5, ls=:dash, label=(r==1 && c==1 ? "Forecast" : ""))

            # Target line (CPI and GDP only)
            if cv.tgt !== nothing
                plot!(fig[sp], cnfctl_dates, cv.tgt,
                    color=:red, lw=1.5, ls=:dot, label=(r==1 && c==1 ? "Target" : ""))
            end

            # 68% posterior bands
            bd_lb = getfield(rr.bands, cv.lb)
            bd_ub = getfield(rr.bands, cv.ub)
            # For NEER, bands are deviations from baseline — add e_base
            if cv.e_adj
                bd_lb = cv.base .+ bd_lb
                bd_ub = cv.base .+ bd_ub
            end
            plot!(fig[sp], cnfctl_dates, bd_lb,
                fillrange=bd_ub, fillalpha=0.2, fillcolor=lblue, lw=0,
                label=(r==1 && c==1 ? "68%" : ""))

            # Counterfactual path
            cnfctl_path = getfield(rr.res, cv.cnfctl)
            if cv.e_adj
                cnfctl_path = cv.base .+ cnfctl_path
            end
            plot!(fig[sp], cnfctl_dates, cnfctl_path,
                color=blue, lw=2.5, label=(r==1 && c==1 ? "Counterfact'l" : ""))

            hline!(fig[sp], [0], color=:gray, ls=:dot, alpha=0.3, label="")

            # Set y-axis limits: use actual data range + counterfactual range, with padding
            all_vals = vcat(
                all_plot_data[:, cv.idx],
                cnfctl_path,
                cv.base,
                bd_lb, bd_ub,
            )
            all_vals = filter(!isnan, all_vals)
            if !isempty(all_vals)
                ylo, yhi = minimum(all_vals), maximum(all_vals)
                pad = 0.15 * max(yhi - ylo, 1.0)
                ylims!(fig[sp], (ylo - pad, yhi + pad))
            end

            # Title on row 1 only; ylabel on column 1 only
            if r == 1
                title!(fig[sp], cv.col_title)
            end
            if c == 1
                ylabel!(fig[sp], rule.label)
            end
        end
    end

    display(fig)
    fname = "cnfctl_$(cfctl_year)_s$(yc.sample).png"
    savefig(fig, joinpath(MAIN_DIR, fname))
    println("\n  Saved: $fname")
end

# =============================================================================
# 6) SAVE
# =============================================================================

serialize(joinpath(intermediate_dir("2025"), "counterfactual_results.jls"), Dict(
    "results" => all_results,
    "rules"   => [(r.label, r.λ_π, r.λ_y, r.λ_i, r.λ_e) for r in rules],
    "year_configs" => [(yc.cfctl_start, yc.cfctl_end, yc.sample) for yc in year_configs],
))
println("\n✓ All counterfactual results saved.")
println("✓ Counterfactual analysis complete.")
