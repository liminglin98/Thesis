##
# =============================================================================
# COUNTERFACTUAL ZERO-DRAW DIAGNOSTICS
# =============================================================================
#
# Usage:
#   julia src/julia/Counterfactual_zero_draw_diagnostics.jl
#
# This standalone diagnostic tests how the HFI zero-draw behavior propagates into
# the 2022 counterfactual figures. It does not modify production artifacts.
# =============================================================================

using LinearAlgebra, Statistics, Printf, Dates, Serialization, Plots
using CSV, DataFrames

include(joinpath(@__DIR__, "common.jl"))

const gdp_idx = 1
const ip_idx = 2
const cpi_idx = 3
const fr007_idx = 4
const neer_idx = 6

const irf_gdp_idx = 1
const irf_cpi_idx = 2
const irf_fr007_idx = 3
const irf_neer_idx = 4
const irf_ip_idx = 5

function forecast_from_date(Y, A_list, c, start_idx, H_fc)
    p_loc = length(A_list)
    history = [Y[start_idx - j + 1, :] for j in 1:p_loc]
    fc = zeros(H_fc, size(Y, 2))
    for h in 1:H_fc
        y_hat = copy(c)
        for j in 1:min(p_loc, length(history))
            y_hat .+= A_list[j] * history[j]
        end
        fc[h, :] = y_hat
        pushfirst!(history, y_hat)
        pop!(history)
    end
    return fc
end

function cpi_target_for_year(yr::Int)
    yr <= 2024 ? 3.0 : 2.0
end

function gdp_target_for_year(yr::Int)
    yr == 2021 ? 6.0 : 5.0
end

function announced_target_year(d::Date)
    month(d) >= 3 ? year(d) : year(d) - 1
end

function cpi_target_for_date(d::Date)
    cpi_target_for_year(announced_target_year(d))
end

function gdp_target_for_date(d::Date)
    gdp_target_for_year(announced_target_year(d))
end

function build_transmission_map(irf_col::AbstractVector{<:Real}, T_hor::Int)
    S = min(T_hor, length(irf_col))
    M = zeros(T_hor, S)
    for s in 1:S, h in s:T_hor
        lag = h - s + 1
        lag <= length(irf_col) && (M[h, s] = irf_col[lag])
    end
    return M
end

function build_two_shock_map(irf_col_1::AbstractVector{<:Real},
                             irf_col_2::AbstractVector{<:Real},
                             T_hor::Int)
    hcat(build_transmission_map(irf_col_1, T_hor),
         build_transmission_map(irf_col_2, T_hor))
end

function build_first_diff_matrix_anchored(T_hor::Int)
    D = zeros(T_hor, T_hor)
    D[1, 1] = 1.0
    for t in 2:T_hor
        D[t, t-1] = -1.0
        D[t, t] = 1.0
    end
    return D
end

function cnfctl_flexible(pi_base, y_base, i_base,
                         Pi_m, Y_m, I_m, E_m, IP_m, ip_base,
                         pi_target, y_target, i_hist_last::Float64,
                         λ_pi, λ_y, λ_i, λ_e)
    T = length(pi_base)
    gap_pi = pi_target .- pi_base
    gap_y = y_target .- y_base
    D_anch = build_first_diff_matrix_anchored(T)
    d_anchor = vcat([i_hist_last], zeros(T - 1))
    gap_i = d_anchor .- D_anch * i_base
    A_i = D_anch * I_m

    A_stack = vcat(sqrt(λ_pi) * Pi_m, sqrt(λ_y) * Y_m,
                   sqrt(λ_i) * A_i, sqrt(λ_e) * E_m)
    b_stack = vcat(sqrt(λ_pi) * gap_pi, sqrt(λ_y) * gap_y,
                   sqrt(λ_i) * gap_i, sqrt(λ_e) * zeros(T))

    nu_tilde = A_stack \ b_stack
    return (
        pi_cnfctl = pi_base .+ Pi_m * nu_tilde,
        y_cnfctl = y_base .+ Y_m * nu_tilde,
        i_cnfctl = i_base .+ I_m * nu_tilde,
        e_cnfctl = E_m * nu_tilde,
        ip_cnfctl = ip_base .+ IP_m * nu_tilde,
    )
end

function nanquantile(X::Matrix{Float64}, q::Float64)
    [let v = filter(!isnan, X[t, :]); isempty(v) ? NaN : quantile(v, q) end
     for t in 1:size(X, 1)]
end

function posterior_bands_flex(pi_base, y_base, i_base, ip_base,
                              narr_draws, hfi_draws, n_draws_irf, fcst_hor,
                              pi_target, y_target, i_hist_last::Float64,
                              rule)
    if n_draws_irf == 0
        return nothing
    end

    pi_d = zeros(fcst_hor, n_draws_irf)
    y_d = zeros(fcst_hor, n_draws_irf)
    i_d = zeros(fcst_hor, n_draws_irf)
    e_d = zeros(fcst_hor, n_draws_irf)
    ip_d = zeros(fcst_hor, n_draws_irf)

    for d in 1:n_draws_irf
        Pi_d = build_two_shock_map(narr_draws[d, :, irf_cpi_idx], hfi_draws[d, :, irf_cpi_idx], fcst_hor)
        Y_d = build_two_shock_map(narr_draws[d, :, irf_gdp_idx], hfi_draws[d, :, irf_gdp_idx], fcst_hor)
        I_d = build_two_shock_map(narr_draws[d, :, irf_fr007_idx], hfi_draws[d, :, irf_fr007_idx], fcst_hor)
        E_d = build_two_shock_map(narr_draws[d, :, irf_neer_idx], hfi_draws[d, :, irf_neer_idx], fcst_hor)
        IP_d = build_two_shock_map(narr_draws[d, :, irf_ip_idx], hfi_draws[d, :, irf_ip_idx], fcst_hor)
        try
            res = cnfctl_flexible(pi_base, y_base, i_base, Pi_d, Y_d, I_d, E_d, IP_d, ip_base,
                                  pi_target, y_target, i_hist_last,
                                  rule.λ_pi, rule.λ_y, rule.λ_i, rule.λ_e)
            pi_d[:, d] = res.pi_cnfctl
            y_d[:, d] = res.y_cnfctl
            i_d[:, d] = res.i_cnfctl
            e_d[:, d] = res.e_cnfctl
            ip_d[:, d] = res.ip_cnfctl
        catch
            pi_d[:, d] .= NaN
            y_d[:, d] .= NaN
            i_d[:, d] .= NaN
            e_d[:, d] .= NaN
            ip_d[:, d] .= NaN
        end
    end

    return (
        pi_med = nanquantile(pi_d, 0.50), pi_lb = nanquantile(pi_d, 0.16), pi_ub = nanquantile(pi_d, 0.84),
        y_med = nanquantile(y_d, 0.50), y_lb = nanquantile(y_d, 0.16), y_ub = nanquantile(y_d, 0.84),
        i_med = nanquantile(i_d, 0.50), i_lb = nanquantile(i_d, 0.16), i_ub = nanquantile(i_d, 0.84),
        e_med = nanquantile(e_d, 0.50), e_lb = nanquantile(e_d, 0.16), e_ub = nanquantile(e_d, 0.84),
        ip_med = nanquantile(ip_d, 0.50), ip_lb = nanquantile(ip_d, 0.16), ip_ub = nanquantile(ip_d, 0.84),
    )
end

function save_counterfactual_figure(setting::String, sample::String, rule_label::String,
                                    hist_dates, hist_data, actual_dates_window, actual_data,
                                    cnfctl_dates, pi_target, y_target,
                                    pi_base, y_base, i_base, e_base, ip_base,
                                    point_res, bands, out_dir)
    blue = RGB(0.45, 0.62, 0.70)
    fig = plot(layout=(1, 5), size=(2200, 420), margin=5Plots.mm,
        plot_title="Counterfactual zero-draw diagnostic | $(setting) | sample $(sample)")

    all_plot_dates = vcat(hist_dates, actual_dates_window)
    all_plot_data = vcat(hist_data, actual_data)
    cols = [
        (idx=cpi_idx, base=pi_base, point=point_res.pi_cnfctl, med=:pi_med, lb=:pi_lb, ub=:pi_ub, tgt=pi_target, title="CPI YoY (%)", add_base=false),
        (idx=gdp_idx, base=y_base, point=point_res.y_cnfctl, med=:y_med, lb=:y_lb, ub=:y_ub, tgt=y_target, title="Real GDP YoY (%)", add_base=false),
        (idx=fr007_idx, base=i_base, point=point_res.i_cnfctl, med=:i_med, lb=:i_lb, ub=:i_ub, tgt=nothing, title="FR007 (%)", add_base=false),
        (idx=neer_idx, base=e_base, point=e_base .+ point_res.e_cnfctl, med=:e_med, lb=:e_lb, ub=:e_ub, tgt=nothing, title="NEER YoY (%)", add_base=true),
        (idx=ip_idx, base=ip_base, point=point_res.ip_cnfctl, med=:ip_med, lb=:ip_lb, ub=:ip_ub, tgt=nothing, title="IP YoY (%)", add_base=false),
    ]

    for (sp, cv) in enumerate(cols)
        plot!(fig[sp], all_plot_dates, all_plot_data[:, cv.idx], color=:black, lw=2, label=(sp == 1 ? "Data" : ""))
        plot!(fig[sp], cnfctl_dates, cv.base, color=:gray, lw=1.5, ls=:dash, label=(sp == 1 ? "Forecast" : ""))
        cv.tgt !== nothing && plot!(fig[sp], cnfctl_dates, cv.tgt, color=:red, lw=1.5, ls=:dot, label=(sp == 1 ? "Target" : ""))

        if bands === nothing
            plot!(fig[sp], cnfctl_dates, cv.point, color=blue, lw=2.5, label=(sp == 1 ? "Point counterfact'l" : ""))
        else
            lb = getfield(bands, cv.lb)
            ub = getfield(bands, cv.ub)
            med = getfield(bands, cv.med)
            if cv.add_base
                lb = cv.base .+ lb
                ub = cv.base .+ ub
                med = cv.base .+ med
            end
            plot!(fig[sp], cnfctl_dates, lb, fillrange=ub, fillalpha=0.2, fillcolor=blue, lw=0, label=(sp == 1 ? "68%" : ""))
            plot!(fig[sp], cnfctl_dates, med, color=blue, lw=2.5, label=(sp == 1 ? "Posterior median" : ""))
            plot!(fig[sp], cnfctl_dates, cv.point, color=:black, lw=1.2, ls=:dashdot, label=(sp == 1 ? "Point" : ""))
        end
        hline!(fig[sp], [0], color=:gray, ls=:dot, alpha=0.3, label="")
        title!(fig[sp], cv.title)
    end

    fpath = joinpath(out_dir, "cnfctl_zero_draw_$(setting)_$(rule_slug(rule_label))_s$(sample).png")
    savefig(fig, fpath)
    println("Saved: ", fpath)
    return fpath
end

function rule_slug(lbl::String)
    s = lowercase(lbl)
    s = replace(s, "+" => "plus")
    s = replace(s, r"[^a-z0-9]+" => "_")
    s = replace(s, r"_+" => "_")
    return strip(s, '_')
end

function save_original_style_figure(setting::String, sample::String, rule_results,
                                    hist_dates, hist_data, actual_dates_window, actual_data,
                                    cnfctl_dates, pi_target, y_target,
                                    pi_base, y_base, i_base, e_base, ip_base,
                                    out_dir)
    blue = RGB(0.45, 0.62, 0.70)
    lblue = RGB(0.45, 0.62, 0.70)
    main_rule_results = [rr for rr in rule_results if rr.rule.label != "More CPI Focused"]
    n_cols = 5
    n_rules = length(main_rule_results)

    fig = plot(layout=(n_rules, n_cols), size=(2200, 250 * n_rules + 50), margin=6Plots.mm,
        plot_title="Counterfactual zero-draw diagnostic from 2023 — $(setting) — sample $(sample)")

    all_plot_dates = vcat(hist_dates, actual_dates_window)
    all_plot_data = vcat(hist_data, actual_data)

    col_vars = [
        (idx=cpi_idx, med=:pi_med, base=pi_base, e_adj=false, lb=:pi_lb, ub=:pi_ub, tgt=pi_target, col_title="CPI YoY (%)"),
        (idx=gdp_idx, med=:y_med, base=y_base, e_adj=false, lb=:y_lb, ub=:y_ub, tgt=y_target, col_title="Real GDP YoY (%)"),
        (idx=fr007_idx, med=:i_med, base=i_base, e_adj=false, lb=:i_lb, ub=:i_ub, tgt=nothing, col_title="FR007 (%)"),
        (idx=neer_idx, med=:e_med, base=e_base, e_adj=true, lb=:e_lb, ub=:e_ub, tgt=nothing, col_title="NEER YoY (%)"),
        (idx=ip_idx, med=:ip_med, base=ip_base, e_adj=false, lb=:ip_lb, ub=:ip_ub, tgt=nothing, col_title="IP YoY (%)"),
    ]

    for (r, rr) in enumerate(main_rule_results)
        for (c, cv) in enumerate(col_vars)
            sp = (r - 1) * n_cols + c
            plot!(fig[sp], all_plot_dates, all_plot_data[:, cv.idx],
                color=:black, lw=2, label=(r == 1 && c == 1 ? "Data" : ""))
            plot!(fig[sp], cnfctl_dates, cv.base,
                color=:gray, lw=1.5, ls=:dash, label=(r == 1 && c == 1 ? "Forecast" : ""))
            if cv.tgt !== nothing
                plot!(fig[sp], cnfctl_dates, cv.tgt,
                    color=:red, lw=1.5, ls=:dot, label=(r == 1 && c == 1 ? "Target" : ""))
            end

            if rr.bands === nothing
                point_path = getfield(rr.point_res, cv.med == :pi_med ? :pi_cnfctl :
                    cv.med == :y_med ? :y_cnfctl :
                    cv.med == :i_med ? :i_cnfctl :
                    cv.med == :e_med ? :e_cnfctl : :ip_cnfctl)
                cv.e_adj && (point_path = cv.base .+ point_path)
                plot!(fig[sp], cnfctl_dates, point_path,
                    color=blue, lw=2.5, label=(r == 1 && c == 1 ? "Point counterfact'l" : ""))
                all_vals = vcat(all_plot_data[:, cv.idx], point_path, cv.base)
            else
                bd_lb = getfield(rr.bands, cv.lb)
                bd_ub = getfield(rr.bands, cv.ub)
                cnfctl_path = getfield(rr.bands, cv.med)
                if cv.e_adj
                    bd_lb = cv.base .+ bd_lb
                    bd_ub = cv.base .+ bd_ub
                    cnfctl_path = cv.base .+ cnfctl_path
                end
                plot!(fig[sp], cnfctl_dates, bd_lb,
                    fillrange=bd_ub, fillalpha=0.2, fillcolor=lblue, lw=0,
                    label=(r == 1 && c == 1 ? "68%" : ""))
                plot!(fig[sp], cnfctl_dates, cnfctl_path,
                    color=blue, lw=2.5, label=(r == 1 && c == 1 ? "Counterfact'l" : ""))
                all_vals = vcat(all_plot_data[:, cv.idx], cnfctl_path, cv.base, bd_lb, bd_ub)
            end

            hline!(fig[sp], [0], color=:gray, ls=:dot, alpha=0.3, label="")
            all_vals = filter(!isnan, all_vals)
            if !isempty(all_vals)
                ylo, yhi = minimum(all_vals), maximum(all_vals)
                pad = 0.15 * max(yhi - ylo, 1.0)
                ylims!(fig[sp], (ylo - pad, yhi + pad))
            end
            r == 1 && title!(fig[sp], cv.col_title)
            c == 1 && ylabel!(fig[sp], rr.rule.label)
        end
    end

    fpath = joinpath(out_dir, "cnfctl_zero_draw_original_style_$(setting)_s$(sample).png")
    savefig(fig, fpath)
    println("Saved: ", fpath)
    return fpath
end

function run_2022_diagnostics()
    sample = "2022"
    cfctl_start = Date(2023, 1, 1)
    cfctl_end = Date(2025, 12, 1)
    rules = [
        (label="CPI Targeting", λ_pi=1.0, λ_y=0.0, λ_i=1.0, λ_e=0.0),
        (label="CPI + GDP Targeting", λ_pi=1.0, λ_y=1.0, λ_i=1.0, λ_e=0.0),
        (label="CPI + GDP + NEER", λ_pi=1.0, λ_y=0.5, λ_i=1.0, λ_e=0.5),
        (label="More CPI Focused", λ_pi=2.0, λ_y=1.0, λ_i=1.0, λ_e=1.0),
    ]
    diagnostic_rule = rules[3]

    inter_dir = intermediate_dir(sample)
    out_dir = diagnostics_dir(sample)

    bvar = deserialize(joinpath(inter_dir, "bvar_results.jls"))
    narr_data = deserialize(joinpath(inter_dir, "narrative_irf_results.jls"))
    hfi_data = deserialize(joinpath(inter_dir, "hfi_irf_ratechange.jls"))

    narr_point = narr_data["irf_point"]
    narr_draws = narr_data["irf_draws"]
    hfi_point = hfi_data["irf_point"]
    hfi_draws_actual = hfi_data["irf_draws"]

    H_irf = min(narr_data["H"], hfi_data["H"])
    narr_point = narr_point[1:H_irf+1, :]
    hfi_point = hfi_point[1:H_irf+1, :]
    narr_draws = narr_draws[:, 1:H_irf+1, :]
    hfi_draws_actual = hfi_draws_actual[:, 1:H_irf+1, :]

    df = CSV.read(joinpath(DERIVED_DIR, "china_longterm_data.csv"), DataFrame)
    df.date = Date.(df.date)
    sort!(df, :date)
    syms = [:realgdp_monthly_yoy, :IP_yoy, :cpi, :FR007, :M2_growth, :neer_yoy, :US_IP_yoy]
    df = dropmissing(df, syms)
    dates_actual = df.date
    Y_actual = Matrix{Float64}(df[:, syms])

    fcst_date_idx = findfirst(d -> d >= cfctl_start, dates_actual)
    fcst_hor = length(cfctl_start:Month(1):cfctl_end)
    cnfctl_dates = [cfctl_start + Month(h - 1) for h in 1:fcst_hor]
    baseline_fc = forecast_from_date(Y_actual, bvar["A_list"], bvar["c"], fcst_date_idx - 1, fcst_hor)

    T_actual_avail = min(fcst_hor, size(Y_actual, 1) - fcst_date_idx + 1)
    actual_data = Y_actual[fcst_date_idx:min(fcst_date_idx + T_actual_avail - 1, end), :]
    actual_dates_window = dates_actual[fcst_date_idx:min(fcst_date_idx + T_actual_avail - 1, end)]
    hist_start_idx = max(1, fcst_date_idx - 12)
    hist_dates = dates_actual[hist_start_idx:fcst_date_idx-1]
    hist_data = Y_actual[hist_start_idx:fcst_date_idx-1, :]
    i_hist_last = hist_data[end, fr007_idx]

    pi_base = baseline_fc[:, cpi_idx]
    y_base = baseline_fc[:, gdp_idx]
    i_base = baseline_fc[:, fr007_idx]
    e_base = baseline_fc[:, neer_idx]
    ip_base = baseline_fc[:, ip_idx]
    pi_target = [cpi_target_for_date(d) for d in cnfctl_dates]
    y_target = [gdp_target_for_date(d) for d in cnfctl_dates]

    Pi_m = build_two_shock_map(narr_point[:, irf_cpi_idx], hfi_point[:, irf_cpi_idx], fcst_hor)
    Y_m = build_two_shock_map(narr_point[:, irf_gdp_idx], hfi_point[:, irf_gdp_idx], fcst_hor)
    I_m = build_two_shock_map(narr_point[:, irf_fr007_idx], hfi_point[:, irf_fr007_idx], fcst_hor)
    E_m = build_two_shock_map(narr_point[:, irf_neer_idx], hfi_point[:, irf_neer_idx], fcst_hor)
    IP_m = build_two_shock_map(narr_point[:, irf_ip_idx], hfi_point[:, irf_ip_idx], fcst_hor)
    point_res = cnfctl_flexible(pi_base, y_base, i_base, Pi_m, Y_m, I_m, E_m, IP_m, ip_base,
                                pi_target, y_target, i_hist_last,
                                diagnostic_rule.λ_pi, diagnostic_rule.λ_y,
                                diagnostic_rule.λ_i, diagnostic_rule.λ_e)

    println("="^78)
    println("Counterfactual zero-draw diagnostics, sample 2022")
    println("Actual HFI draws: ", size(hfi_draws_actual))
    println("Single-rule diagnostic graph uses: ", diagnostic_rule.label,
        " with lambdas = ", (diagnostic_rule.λ_pi, diagnostic_rule.λ_y, diagnostic_rule.λ_i, diagnostic_rule.λ_e))
    println("="^78)

    n_actual = min(size(narr_draws, 1), size(hfi_draws_actual, 1))
    bands_current_actual = posterior_bands_flex(pi_base, y_base, i_base, ip_base,
        narr_draws, hfi_draws_actual, n_actual, fcst_hor,
        pi_target, y_target, i_hist_last, diagnostic_rule)
    save_counterfactual_figure("current_actual", sample, diagnostic_rule.label,
        hist_dates, hist_data, actual_dates_window, actual_data, cnfctl_dates,
        pi_target, y_target, pi_base, y_base, i_base, e_base, ip_base,
        point_res, bands_current_actual, out_dir)

    bands_adjusted_actual = posterior_bands_flex(pi_base, y_base, i_base, ip_base,
        narr_draws, hfi_draws_actual, n_actual, fcst_hor,
        pi_target, y_target, i_hist_last, diagnostic_rule)
    save_counterfactual_figure("adjusted_actual", sample, diagnostic_rule.label,
        hist_dates, hist_data, actual_dates_window, actual_data, cnfctl_dates,
        pi_target, y_target, pi_base, y_base, i_base, e_base, ip_base,
        point_res, bands_adjusted_actual, out_dir)

    hfi_draws_current_forced = zeros(1, H_irf + 1, size(hfi_point, 2))
    bands_current_forced = posterior_bands_flex(pi_base, y_base, i_base, ip_base,
        narr_draws, hfi_draws_current_forced, 1, fcst_hor,
        pi_target, y_target, i_hist_last, diagnostic_rule)
    save_counterfactual_figure("current_forced_zero_placeholder", sample, diagnostic_rule.label,
        hist_dates, hist_data, actual_dates_window, actual_data, cnfctl_dates,
        pi_target, y_target, pi_base, y_base, i_base, e_base, ip_base,
        point_res, bands_current_forced, out_dir)

    bands_adjusted_forced = posterior_bands_flex(pi_base, y_base, i_base, ip_base,
        narr_draws, hfi_draws_current_forced[1:0, :, :], 0, fcst_hor,
        pi_target, y_target, i_hist_last, diagnostic_rule)
    save_counterfactual_figure("adjusted_forced_zero_point_only", sample, diagnostic_rule.label,
        hist_dates, hist_data, actual_dates_window, actual_data, cnfctl_dates,
        pi_target, y_target, pi_base, y_base, i_base, e_base, ip_base,
        point_res, bands_adjusted_forced, out_dir)

    for (setting, hfi_draws, ndraws) in [
        ("current_actual", hfi_draws_actual, n_actual),
        ("adjusted_actual", hfi_draws_actual, n_actual),
        ("current_forced_zero_placeholder", hfi_draws_current_forced, 1),
        ("adjusted_forced_zero_point_only", hfi_draws_current_forced[1:0, :, :], 0),
    ]
        rule_results = []
        for rule in rules
            point_res_rule = cnfctl_flexible(pi_base, y_base, i_base, Pi_m, Y_m, I_m, E_m, IP_m, ip_base,
                pi_target, y_target, i_hist_last,
                rule.λ_pi, rule.λ_y, rule.λ_i, rule.λ_e)
            bands_rule = posterior_bands_flex(pi_base, y_base, i_base, ip_base,
                narr_draws, hfi_draws, ndraws, fcst_hor,
                pi_target, y_target, i_hist_last, rule)
            push!(rule_results, (rule=rule, point_res=point_res_rule, bands=bands_rule))
        end
        save_original_style_figure(setting, sample, rule_results,
            hist_dates, hist_data, actual_dates_window, actual_data,
            cnfctl_dates, pi_target, y_target,
            pi_base, y_base, i_base, e_base, ip_base, out_dir)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_2022_diagnostics()
end
