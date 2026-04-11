##
# =============================================================================
# COUNTERFACTUAL HISTORICAL SCENARIOS — LOSS FUNCTION APPROACH
# =============================================================================
#
# Following Wolf et al. (2025) — loss function counterfactual
#
# Method: minimize weighted loss L with anchored FR007 smoothing, solved in closed form.
#   L = λ_π ‖π - π*‖² + λ_y ‖y - y*‖² + λ_i ‖D_anch i - d_anchor‖² + λ_e ‖E_m ν‖²
#
# Three targeting rules (all include anchored FR007 smoothing):
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
const ip_idx    = 2
const cpi_idx   = 3
const fr007_idx = 4
const neer_idx  = 6

# IRF ordering (RRShocks/HFIShocks): 5 variables
#   1: GDP   2: CPI   3: FR007   4: NEER   5: IP
const irf_gdp_idx   = 1
const irf_cpi_idx   = 2
const irf_fr007_idx = 3
const irf_neer_idx  = 4
const irf_ip_idx    = 5

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

function build_first_diff_matrix_anchored(T_hor::Int)
    # T x T lower-bidiagonal: first row anchors to last historical FR007.
    D = zeros(T_hor, T_hor)
    D[1, 1] = 1.0
    for t in 2:T_hor
        D[t, t-1] = -1.0
        D[t, t]   = 1.0
    end
    return D
end

function cnfctl_flexible(pi_base, y_base, i_base,
                         Pi_m, Y_m, I_m, E_m, IP_m, ip_base,
                         pi_target, y_target, i_hist_last::Float64,
                         λ_pi, λ_y, λ_i, λ_e)
    T = length(pi_base)
    gap_pi = pi_target .- pi_base
    gap_y  = y_target  .- y_base

    D_anch   = build_first_diff_matrix_anchored(T)
    DI       = D_anch * I_m
    Di_base  = D_anch * i_base
    d_anchor = zeros(T); d_anchor[1] = i_hist_last

    sq_pi = sqrt(λ_pi); sq_y = sqrt(λ_y); sq_i = sqrt(λ_i); sq_e = sqrt(λ_e)

    A_stack = vcat(sq_pi * Pi_m, sq_y * Y_m, sq_i * DI, sq_e * E_m)
    b_stack = vcat(sq_pi * gap_pi, sq_y * gap_y, sq_i * (d_anchor .- Di_base), sq_e * zeros(T))

    nu_tilde  = A_stack \ b_stack
    pi_cnfctl = pi_base .+ Pi_m * nu_tilde
    y_cnfctl  = y_base  .+ Y_m  * nu_tilde
    i_cnfctl  = i_base  .+ I_m  * nu_tilde
    e_cnfctl  = E_m * nu_tilde
    ip_cnfctl = ip_base .+ IP_m * nu_tilde

    L_pi = λ_pi * sum((pi_cnfctl .- pi_target).^2)
    L_y  = λ_y  * sum((y_cnfctl  .- y_target).^2)
    L_i  = λ_i  * sum((D_anch * i_cnfctl .- d_anchor).^2)
    L_e  = λ_e  * sum(e_cnfctl.^2)

    return (pi_cnfctl=pi_cnfctl, y_cnfctl=y_cnfctl, i_cnfctl=i_cnfctl,
            e_cnfctl=e_cnfctl, ip_cnfctl=ip_cnfctl, nu_tilde=nu_tilde,
            loss=Dict("total"=>L_pi+L_y+L_i+L_e, "pi"=>L_pi, "y"=>L_y, "i"=>L_i, "e"=>L_e))
end

function shift_left_with_zero(v::Vector{Float64})
    T = length(v)
    T == 1 && return [0.0]
    return vcat(v[2:end], 0.0)
end

function cnfctl_flexible_evol(Y, A_list, c,
                              narr_point, hfi_point,
                              fcst_date_idx::Int, fcst_hor::Int,
                              pi_target::Vector{Float64}, y_target::Vector{Float64},
                              λ_pi, λ_y, λ_i, λ_e)
    pi_cf = zeros(fcst_hor)
    y_cf  = zeros(fcst_hor)
    i_cf  = zeros(fcst_hor)
    e_cf  = zeros(fcst_hor)
    ip_cf = zeros(fcst_hor)

    pi_base_roll = zeros(fcst_hor)
    y_base_roll  = zeros(fcst_hor)
    i_base_roll  = zeros(fcst_hor)
    e_base_roll  = zeros(fcst_hor)
    ip_base_roll = zeros(fcst_hor)

    L_pi = 0.0; L_y = 0.0; L_i = 0.0; L_e = 0.0

    prev_pi_base = Float64[]
    prev_y_base  = Float64[]

    max_roll = min(fcst_hor, size(Y, 1) - fcst_date_idx + 1)
    last_res = nothing
    last_base = nothing

    for t_simul in 1:max_roll
        start_idx_t = fcst_date_idx + t_simul - 1
        H_t = fcst_hor - t_simul + 1

        baseline_t = forecast_from_date(Y, A_list, c, start_idx_t - 1, H_t)
        pi_base_t = baseline_t[:, cpi_idx]
        y_base_t  = baseline_t[:, gdp_idx]
        i_base_t  = baseline_t[:, fr007_idx]
        e_base_t  = baseline_t[:, neer_idx]
        ip_base_t = baseline_t[:, ip_idx]

        Pi_t = build_two_shock_map(narr_point[:, irf_cpi_idx],   hfi_point[:, irf_cpi_idx],   H_t)
        Y_t  = build_two_shock_map(narr_point[:, irf_gdp_idx],   hfi_point[:, irf_gdp_idx],   H_t)
        I_t  = build_two_shock_map(narr_point[:, irf_fr007_idx], hfi_point[:, irf_fr007_idx], H_t)
        E_t  = build_two_shock_map(narr_point[:, irf_neer_idx],  hfi_point[:, irf_neer_idx],  H_t)
        IP_t = build_two_shock_map(narr_point[:, irf_ip_idx],    hfi_point[:, irf_ip_idx],    H_t)

        pi_target_t = pi_target[t_simul:end]
        y_target_t  = y_target[t_simul:end]
        if t_simul > 1
            # Forecast-revision gap: current baseline minus last-step baseline shifted by one month.
            pi_gap_t = pi_base_t .- prev_pi_base[2:end]
            y_gap_t  = y_base_t  .- prev_y_base[2:end]
            pi_target_t = pi_base_t .+ pi_gap_t
            y_target_t  = y_base_t  .+ y_gap_t
        end

        i_hist_last_t = Y[start_idx_t - 1, fr007_idx]
        res_t = cnfctl_flexible(pi_base_t, y_base_t, i_base_t,
                                Pi_t, Y_t, I_t, E_t, IP_t, ip_base_t,
                                pi_target_t, y_target_t, i_hist_last_t,
                                λ_pi, λ_y, λ_i, λ_e)
        last_res = res_t
        last_base = (pi=pi_base_t, y=y_base_t, i=i_base_t, e=e_base_t, ip=ip_base_t)

        pi_cf[t_simul] = res_t.pi_cnfctl[1]
        y_cf[t_simul]  = res_t.y_cnfctl[1]
        i_cf[t_simul]  = res_t.i_cnfctl[1]
        e_cf[t_simul]  = res_t.e_cnfctl[1]
        ip_cf[t_simul] = res_t.ip_cnfctl[1]

        pi_base_roll[t_simul] = pi_base_t[1]
        y_base_roll[t_simul]  = y_base_t[1]
        i_base_roll[t_simul]  = i_base_t[1]
        e_base_roll[t_simul]  = e_base_t[1]
        ip_base_roll[t_simul] = ip_base_t[1]

        L_pi += res_t.loss["pi"]
        L_y  += res_t.loss["y"]
        L_i  += res_t.loss["i"]
        L_e  += res_t.loss["e"]

        prev_pi_base = pi_base_t
        prev_y_base  = y_base_t
    end

    if max_roll < fcst_hor && last_res !== nothing
        rem = fcst_hor - max_roll
        pi_cf[max_roll+1:end] = last_res.pi_cnfctl[2:rem+1]
        y_cf[max_roll+1:end]  = last_res.y_cnfctl[2:rem+1]
        i_cf[max_roll+1:end]  = last_res.i_cnfctl[2:rem+1]
        e_cf[max_roll+1:end]  = last_res.e_cnfctl[2:rem+1]
        ip_cf[max_roll+1:end] = last_res.ip_cnfctl[2:rem+1]

        pi_base_roll[max_roll+1:end] = last_base.pi[2:rem+1]
        y_base_roll[max_roll+1:end]  = last_base.y[2:rem+1]
        i_base_roll[max_roll+1:end]  = last_base.i[2:rem+1]
        e_base_roll[max_roll+1:end]  = last_base.e[2:rem+1]
        ip_base_roll[max_roll+1:end] = last_base.ip[2:rem+1]
    end

    return (
        pi_cnfctl=pi_cf, y_cnfctl=y_cf, i_cnfctl=i_cf, e_cnfctl=e_cf, ip_cnfctl=ip_cf,
        pi_base_roll=pi_base_roll, y_base_roll=y_base_roll, i_base_roll=i_base_roll,
        e_base_roll=e_base_roll, ip_base_roll=ip_base_roll,
        loss=Dict("total"=>L_pi+L_y+L_i+L_e, "pi"=>L_pi, "y"=>L_y, "i"=>L_i, "e"=>L_e),
    )
end

function rule_slug(lbl::String)
    s = lowercase(lbl)
    s = replace(s, "+" => "plus")
    s = replace(s, r"[^a-z0-9]+" => "_")
    s = replace(s, r"_+" => "_")
    return strip(s, '_')
end

function save_rule_figure(method::String, sample::String, rule_label::String,
                          hist_dates, hist_data, actual_dates_window, actual_data,
                          cnfctl_dates, pi_target, y_target,
                          pi_base, y_base, i_base, e_base, ip_base,
                          pi_cf, y_cf, i_cf, e_cf, ip_cf,
                          out_dir)
    blue  = RGB(0.45, 0.62, 0.70)
    fig = plot(layout=(1, 5), size=(2200, 420), margin=5Plots.mm,
        plot_title="$(uppercase(method)) | $(rule_label) | sample $(sample)")

    all_plot_dates = vcat(hist_dates, actual_dates_window)
    all_plot_data  = vcat(hist_data, actual_data)

    col_vars = [
        (idx=cpi_idx,  base=pi_base, cf=pi_cf, tgt=pi_target, ctitle="CPI YoY (%)"),
        (idx=gdp_idx,  base=y_base,  cf=y_cf,  tgt=y_target,  ctitle="Real GDP YoY (%)"),
        (idx=ip_idx,   base=ip_base, cf=ip_cf, tgt=nothing,   ctitle="IP YoY (%)"),
        (idx=fr007_idx,base=i_base,  cf=i_cf,  tgt=nothing,   ctitle="FR007 (%)"),
        (idx=neer_idx, base=e_base,  cf=(e_base .+ e_cf), tgt=nothing, ctitle="NEER YoY (%)"),
    ]

    for (sp, cv) in enumerate(col_vars)
        plot!(fig[sp], all_plot_dates, all_plot_data[:, cv.idx], color=:black, lw=2, label=(sp==1 ? "Data" : ""))
        plot!(fig[sp], cnfctl_dates, cv.base, color=:gray, lw=1.5, ls=:dash, label=(sp==1 ? "Forecast" : ""))
        if cv.tgt !== nothing
            plot!(fig[sp], cnfctl_dates, cv.tgt, color=:red, lw=1.5, ls=:dot, label=(sp==1 ? "Target" : ""))
        end
        plot!(fig[sp], cnfctl_dates, cv.cf, color=blue, lw=2.5, label=(sp==1 ? "Counterfact'l" : ""))
        hline!(fig[sp], [0], color=:gray, ls=:dot, alpha=0.3, label="")
        title!(fig[sp], cv.ctitle)
    end

    fpath = joinpath(out_dir, "cnfctl_$(method)_$(rule_slug(rule_label))_s$(sample).png")
    savefig(fig, fpath)
    return fpath
end

function nanquantile(X::Matrix{Float64}, q::Float64)
    [let v = filter(!isnan, X[t,:]); isempty(v) ? NaN : quantile(v, q) end
     for t in 1:size(X,1)]
end

function posterior_bands_flex(pi_base, y_base, i_base, ip_base,
                              narr_draws, hfi_draws, n_draws_irf, fcst_hor,
                              pi_target, y_target, i_hist_last::Float64,
                              λ_pi, λ_y, λ_i, λ_e)
    pi_d = zeros(fcst_hor, n_draws_irf)
    y_d  = zeros(fcst_hor, n_draws_irf)
    i_d  = zeros(fcst_hor, n_draws_irf)
    e_d  = zeros(fcst_hor, n_draws_irf)
    ip_d = zeros(fcst_hor, n_draws_irf)
    nv = 0
    for d in 1:n_draws_irf
        Pi_d  = build_two_shock_map(narr_draws[d,:,irf_cpi_idx],   hfi_draws[d,:,irf_cpi_idx],   fcst_hor)
        Y_d   = build_two_shock_map(narr_draws[d,:,irf_gdp_idx],   hfi_draws[d,:,irf_gdp_idx],   fcst_hor)
        I_d   = build_two_shock_map(narr_draws[d,:,irf_fr007_idx], hfi_draws[d,:,irf_fr007_idx], fcst_hor)
        E_d   = build_two_shock_map(narr_draws[d,:,irf_neer_idx],  hfi_draws[d,:,irf_neer_idx],  fcst_hor)
        IP_d  = build_two_shock_map(narr_draws[d,:,irf_ip_idx],    hfi_draws[d,:,irf_ip_idx],    fcst_hor)
        try
            res = cnfctl_flexible(pi_base, y_base, i_base, Pi_d, Y_d, I_d, E_d, IP_d, ip_base,
                                  pi_target, y_target, i_hist_last, λ_pi, λ_y, λ_i, λ_e)
            pi_d[:,d] = res.pi_cnfctl; y_d[:,d] = res.y_cnfctl
            i_d[:,d]  = res.i_cnfctl;  e_d[:,d] = res.e_cnfctl
            ip_d[:,d] = res.ip_cnfctl
            nv += 1
        catch
            pi_d[:,d] .= NaN; y_d[:,d] .= NaN; i_d[:,d] .= NaN
            e_d[:,d]  .= NaN; ip_d[:,d] .= NaN
        end
    end
    println("  Valid draws: $nv / $n_draws_irf")
    return (;
        pi_lb = nanquantile(pi_d, 0.16), pi_ub = nanquantile(pi_d, 0.84),
        y_lb  = nanquantile(y_d,  0.16), y_ub  = nanquantile(y_d,  0.84),
        i_lb  = nanquantile(i_d,  0.16), i_ub  = nanquantile(i_d,  0.84),
        e_lb  = nanquantile(e_d,  0.16), e_ub  = nanquantile(e_d,  0.84),
        ip_lb = nanquantile(ip_d, 0.16), ip_ub = nanquantile(ip_d, 0.84),
    )
end

function posterior_bands_flex_evol(Y, A_list, c,
                                   narr_draws, hfi_draws, n_draws_irf,
                                   fcst_date_idx::Int, fcst_hor::Int,
                                   pi_target, y_target,
                                   λ_pi, λ_y, λ_i, λ_e)
    pi_d = zeros(fcst_hor, n_draws_irf)
    y_d  = zeros(fcst_hor, n_draws_irf)
    i_d  = zeros(fcst_hor, n_draws_irf)
    e_d  = zeros(fcst_hor, n_draws_irf)
    ip_d = zeros(fcst_hor, n_draws_irf)
    nv = 0
    for d in 1:n_draws_irf
        narr_point_d = Matrix(narr_draws[d, :, :])
        hfi_point_d  = Matrix(hfi_draws[d, :, :])
        try
            res = cnfctl_flexible_evol(Y, A_list, c,
                                       narr_point_d, hfi_point_d,
                                       fcst_date_idx, fcst_hor,
                                       pi_target, y_target,
                                       λ_pi, λ_y, λ_i, λ_e)
            pi_d[:,d] = res.pi_cnfctl
            y_d[:,d]  = res.y_cnfctl
            i_d[:,d]  = res.i_cnfctl
            e_d[:,d]  = res.e_cnfctl
            ip_d[:,d] = res.ip_cnfctl
            nv += 1
        catch
            pi_d[:,d] .= NaN; y_d[:,d] .= NaN; i_d[:,d] .= NaN
            e_d[:,d]  .= NaN; ip_d[:,d] .= NaN
        end
    end
    println("  Valid evol draws: $nv / $n_draws_irf")
    return (;
        pi_lb = nanquantile(pi_d, 0.16), pi_ub = nanquantile(pi_d, 0.84),
        y_lb  = nanquantile(y_d,  0.16), y_ub  = nanquantile(y_d,  0.84),
        i_lb  = nanquantile(i_d,  0.16), i_ub  = nanquantile(i_d,  0.84),
        e_lb  = nanquantile(e_d,  0.16), e_ub  = nanquantile(e_d,  0.84),
        ip_lb = nanquantile(ip_d, 0.16), ip_ub = nanquantile(ip_d, 0.84),
    )
end

# =============================================================================
# 4) CONFIGURATION
# =============================================================================

# Year configurations: (counterfactual start, end, estimation sample label)
year_configs = [
    (cfctl_start=Date(2023,1,1), cfctl_end=Date(2025,12,1), sample="2022", n_hist=12),
    (cfctl_start=Date(2023,1,1), cfctl_end=Date(2025,12,1), sample="2025", n_hist=12),
]

# Targeting rules (rows in each figure): all include FR007 impact-jump smoothing
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
    i_hist_last = hist_data[end, fr007_idx]

    pi_base = baseline_fc[:, cpi_idx]
    y_base  = baseline_fc[:, gdp_idx]
    i_base  = baseline_fc[:, fr007_idx]
    e_base  = baseline_fc[:, neer_idx]
    ip_base = baseline_fc[:, ip_idx]
    println("  i_hist_last = $(i_hist_last),  i_base[1] = $(i_base[1]),  gap = $(i_base[1] - i_hist_last)")
    pi_target = [cpi_target_for_year(year(d)) for d in cnfctl_dates]
    y_target  = [gdp_target_for_year(year(d)) for d in cnfctl_dates]

    Pi_m = build_two_shock_map(narr_point[:, irf_cpi_idx],   hfi_point[:, irf_cpi_idx],   fcst_hor)
    Y_m  = build_two_shock_map(narr_point[:, irf_gdp_idx],   hfi_point[:, irf_gdp_idx],   fcst_hor)
    I_m  = build_two_shock_map(narr_point[:, irf_fr007_idx], hfi_point[:, irf_fr007_idx], fcst_hor)
    E_m  = build_two_shock_map(narr_point[:, irf_neer_idx],  hfi_point[:, irf_neer_idx],  fcst_hor)
    IP_m = build_two_shock_map(narr_point[:, irf_ip_idx],    hfi_point[:, irf_ip_idx],    fcst_hor)

    println("  Horizon: $fcst_hor months, maps: $(size(Pi_m))")

    # --- Solve all 3 rules, collect scenario/evol results ---
    cfctl_year = string(year(yc.cfctl_start))
    rule_results = []  # Vector of (rule, scenario_res, scenario_bands, evol_res, evol_bands)

    for rule in rules
        key_scenario = "scenario_$(yc.sample)_$(cfctl_year)_$(rule.label)"
        key_evol = "evol_$(yc.sample)_$(cfctl_year)_$(rule.label)"

        println("\n", "#"^60)
        println("# $(rule.label)  |  cfctl $(cfctl_year) → sample $(yc.sample) [scenario]")
        println("#"^60)

        res = cnfctl_flexible(pi_base, y_base, i_base, Pi_m, Y_m, I_m, E_m, IP_m, ip_base,
                              pi_target, y_target, i_hist_last,
                              rule.λ_π, rule.λ_y, rule.λ_i, rule.λ_e)

        println(@sprintf("  Loss: total=%.1f  π=%.1f  y=%.1f  jump=%.1f  e=%.1f",
            res.loss["total"], res.loss["pi"], res.loss["y"], res.loss["i"], res.loss["e"]))
        println(@sprintf("  CPI:   mean=%.2f%%  range=[%.2f, %.2f]",
            mean(res.pi_cnfctl), minimum(res.pi_cnfctl), maximum(res.pi_cnfctl)))
        println(@sprintf("  FR007: mean=%.2f%%  range=[%.2f, %.2f]",
            mean(res.i_cnfctl), minimum(res.i_cnfctl), maximum(res.i_cnfctl)))

        println("  Computing posterior bands...")
        bands = posterior_bands_flex(pi_base, y_base, i_base, ip_base,
                    narr_draws, hfi_draws, n_draws_irf, fcst_hor,
                    pi_target, y_target, i_hist_last,
                    rule.λ_π, rule.λ_y, rule.λ_i, rule.λ_e)

        println("\n", "#"^60)
        println("# $(rule.label)  |  cfctl $(cfctl_year) → sample $(yc.sample) [evol]")
        println("#"^60)
        evol_res = cnfctl_flexible_evol(Y_actual, A_list, c_vec,
                        narr_point, hfi_point,
                        fcst_date_idx, fcst_hor,
                        pi_target, y_target,
                        rule.λ_π, rule.λ_y, rule.λ_i, rule.λ_e)

        println(@sprintf("  Loss: total=%.1f  π=%.1f  y=%.1f  jump=%.1f  e=%.1f",
            evol_res.loss["total"], evol_res.loss["pi"], evol_res.loss["y"], evol_res.loss["i"], evol_res.loss["e"]))
        println(@sprintf("  CPI:   mean=%.2f%%  range=[%.2f, %.2f]",
            mean(evol_res.pi_cnfctl), minimum(evol_res.pi_cnfctl), maximum(evol_res.pi_cnfctl)))
        println(@sprintf("  FR007: mean=%.2f%%  range=[%.2f, %.2f]",
            mean(evol_res.i_cnfctl), minimum(evol_res.i_cnfctl), maximum(evol_res.i_cnfctl)))

        n_draws_evol_bands = min(n_draws_irf, 300)
        println("  Computing evol posterior bands... (draws=$(n_draws_evol_bands))")
        evol_bands = posterior_bands_flex_evol(Y_actual, A_list, c_vec,
                narr_draws, hfi_draws, n_draws_evol_bands,
                        fcst_date_idx, fcst_hor,
                        pi_target, y_target,
                        rule.λ_π, rule.λ_y, rule.λ_i, rule.λ_e)

        push!(rule_results, (rule=rule, scenario_res=res, scenario_bands=bands, evol_res=evol_res, evol_bands=evol_bands))

        all_results[key_scenario] = Dict(
            "cfctl_start" => yc.cfctl_start, "sample" => yc.sample,
            "variant" => "scenario",
            "rule" => rule.label, "lambdas" => (rule.λ_π, rule.λ_y, rule.λ_i, rule.λ_e),
            "cnfctl_dates" => cnfctl_dates,
            "res" => res, "bands" => bands,
            "baseline_fc" => baseline_fc,
            "pi_target" => pi_target, "y_target" => y_target,
        )

        all_results[key_evol] = Dict(
            "cfctl_start" => yc.cfctl_start, "sample" => yc.sample,
            "variant" => "evol",
            "rule" => rule.label, "lambdas" => (rule.λ_π, rule.λ_y, rule.λ_i, rule.λ_e),
            "cnfctl_dates" => cnfctl_dates,
            "res" => evol_res, "bands" => evol_bands,
            "baseline_fc" => baseline_fc,
            "pi_target" => pi_target, "y_target" => y_target,
        )

    end

    # --- Build combined 3×5 figure (rows=rules, cols=CPI/GDP/IP/FR007/NEER) ---
    blue  = RGB(0.45, 0.62, 0.70)
    lblue = RGB(0.45, 0.62, 0.70)
    n_cols = 5

    fig = plot(layout=(3, n_cols), size=(2200, 1000), margin=6Plots.mm,
        plot_title="Counterfactual from $(cfctl_year) — sample $(yc.sample)")

    # Continuous actual data line: history + counterfactual window (no gap)
    all_plot_dates = vcat(hist_dates, actual_dates_window)
    all_plot_data  = vcat(hist_data, actual_data)

    # Columns: CPI, GDP, IP, FR007, NEER
    col_vars = [
        (idx=cpi_idx,   cnfctl=:pi_cnfctl, base=pi_base, e_adj=false, lb=:pi_lb, ub=:pi_ub, tgt=pi_target,  col_title="CPI YoY (%)"),
        (idx=gdp_idx,   cnfctl=:y_cnfctl,  base=y_base,  e_adj=false, lb=:y_lb,  ub=:y_ub,  tgt=y_target,   col_title="Real GDP YoY (%)"),
        (idx=ip_idx,    cnfctl=:ip_cnfctl, base=ip_base, e_adj=false, lb=:ip_lb, ub=:ip_ub, tgt=nothing,    col_title="IP YoY (%)"),
        (idx=fr007_idx, cnfctl=:i_cnfctl,  base=i_base,  e_adj=false, lb=:i_lb,  ub=:i_ub,  tgt=nothing,    col_title="FR007 (%)"),
        (idx=neer_idx,  cnfctl=:e_cnfctl,  base=e_base,  e_adj=true,  lb=:e_lb,  ub=:e_ub,  tgt=nothing,    col_title="NEER YoY (%)"),
    ]

    for (r, rr) in enumerate(rule_results)
        rule = rr.rule
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
            bd_lb = getfield(rr.scenario_bands, cv.lb)
            bd_ub = getfield(rr.scenario_bands, cv.ub)
            # For NEER, bands are deviations from baseline — add e_base
            if cv.e_adj
                bd_lb = cv.base .+ bd_lb
                bd_ub = cv.base .+ bd_ub
            end
            plot!(fig[sp], cnfctl_dates, bd_lb,
                fillrange=bd_ub, fillalpha=0.2, fillcolor=lblue, lw=0,
                label=(r==1 && c==1 ? "68%" : ""))

            # Counterfactual path
            cnfctl_path = getfield(rr.scenario_res, cv.cnfctl)
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
    fname_scenario = "cnfctl_scenario_$(cfctl_year)_s$(yc.sample).png"
    savefig(fig, joinpath(MAIN_DIR, fname_scenario))
    println("\n  Saved: $fname_scenario")

    # --- Build combined 3x5 figure for evol (rows=rules, cols=CPI/GDP/IP/FR007/NEER) ---
    fig_evol = plot(layout=(3, n_cols), size=(2200, 1000), margin=6Plots.mm,
        plot_title="Counterfactual EVOL from $(cfctl_year) — sample $(yc.sample)")

    for (r, rr) in enumerate(rule_results)
        rule = rr.rule
        for (c, cv) in enumerate(col_vars)
            sp = (r - 1) * n_cols + c

            # Actual data — one continuous line
            plot!(fig_evol[sp], all_plot_dates, all_plot_data[:, cv.idx],
                color=:black, lw=2, label=(r==1 && c==1 ? "Data" : ""))

            # Evol rolling baseline forecast
            if cv.idx == cpi_idx
                base_evol = rr.evol_res.pi_base_roll
            elseif cv.idx == gdp_idx
                base_evol = rr.evol_res.y_base_roll
            elseif cv.idx == ip_idx
                base_evol = rr.evol_res.ip_base_roll
            elseif cv.idx == fr007_idx
                base_evol = rr.evol_res.i_base_roll
            else
                base_evol = rr.evol_res.e_base_roll
            end
            plot!(fig_evol[sp], cnfctl_dates, base_evol,
                color=:gray, lw=1.5, ls=:dash, label=(r==1 && c==1 ? "Forecast" : ""))

            # Target line (CPI and GDP only)
            if cv.tgt !== nothing
                plot!(fig_evol[sp], cnfctl_dates, cv.tgt,
                    color=:red, lw=1.5, ls=:dot, label=(r==1 && c==1 ? "Target" : ""))
            end

            # 68% posterior bands for evol
            bd_lb = getfield(rr.evol_bands, cv.lb)
            bd_ub = getfield(rr.evol_bands, cv.ub)
            if cv.e_adj
                bd_lb = base_evol .+ bd_lb
                bd_ub = base_evol .+ bd_ub
            end
            plot!(fig_evol[sp], cnfctl_dates, bd_lb,
                fillrange=bd_ub, fillalpha=0.2, fillcolor=lblue, lw=0,
                label=(r==1 && c==1 ? "68%" : ""))

            # Evol counterfactual path
            cnfctl_path = getfield(rr.evol_res, cv.cnfctl)
            if cv.e_adj
                cnfctl_path = base_evol .+ cnfctl_path
            end
            plot!(fig_evol[sp], cnfctl_dates, cnfctl_path,
                color=blue, lw=2.5, label=(r==1 && c==1 ? "Counterfact'l" : ""))

            hline!(fig_evol[sp], [0], color=:gray, ls=:dot, alpha=0.3, label="")

            # Set y-axis limits: use actual data range + counterfactual range, with padding
            all_vals = vcat(
                all_plot_data[:, cv.idx],
                cnfctl_path,
                base_evol,
                bd_lb, bd_ub,
            )
            all_vals = filter(!isnan, all_vals)
            if !isempty(all_vals)
                ylo, yhi = minimum(all_vals), maximum(all_vals)
                pad = 0.15 * max(yhi - ylo, 1.0)
                ylims!(fig_evol[sp], (ylo - pad, yhi + pad))
            end

            # Title on row 1 only; ylabel on column 1 only
            if r == 1
                title!(fig_evol[sp], cv.col_title)
            end
            if c == 1
                ylabel!(fig_evol[sp], rule.label)
            end
        end
    end

    display(fig_evol)
    fname_evol = "cnfctl_evol_$(cfctl_year)_s$(yc.sample).png"
    savefig(fig_evol, joinpath(MAIN_DIR, fname_evol))
    println("  Saved: $fname_evol")
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
