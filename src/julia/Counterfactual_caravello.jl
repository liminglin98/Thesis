using LinearAlgebra, Dates, Serialization, CSV, DataFrames, Printf, Plots

include(joinpath(@__DIR__, "common.jl"))

# Variable ordering follows Counterfactual.jl
const gdp_idx = 1
const ip_idx = 2
const cpi_idx = 3
const fr007_idx = 4
const neer_idx = 6

# IRF ordering (RR/HFI): GDP, CPI, FR007, NEER, IP
const irf_gdp_idx = 1
const irf_cpi_idx = 2
const irf_fr007_idx = 3
const irf_neer_idx = 4
const irf_ip_idx = 5

function cpi_target_for_year(yr::Int)
    yr <= 2024 ? 3.0 : 2.0
end

function gdp_target_for_year(yr::Int)
    yr == 2021 ? 6.0 : 5.0
end

function forecast_from_date(Y, A_list, c, start_idx, H_fc)
    p_loc = length(A_list)
    n_loc = size(Y, 2)
    history = [Y[start_idx - j + 1, :] for j in 1:p_loc]
    fc = zeros(H_fc, n_loc)

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

function build_transmission_map(irf_col::AbstractVector{<:Real}, T_hor::Int)
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

function build_two_shock_map(irf_col_1::AbstractVector{<:Real}, irf_col_2::AbstractVector{<:Real}, T_hor::Int)
    return hcat(build_transmission_map(irf_col_1, T_hor), build_transmission_map(irf_col_2, T_hor))
end

function _caravello_system(Pi_m, Y_m, I_m, W_pi, W_y, W_i)
    A_pi = Pi_m' * W_pi
    A_y = Y_m' * W_y
    A_i = I_m' * W_i
    lhs = A_pi * Pi_m + A_y * Y_m + A_i * I_m
    return (A_pi=A_pi, A_y=A_y, A_i=A_i, lhs=lhs)
end

function _caravello_solve(sys, Pi_m, Y_m, I_m, pi_x, y_x, i_x, wedge)
    rhs = sys.A_pi * pi_x + sys.A_y * y_x + sys.A_i * i_x - wedge
    d = sys.lhs \ rhs
    pi_path = pi_x - Pi_m * d
    y_path = y_x - Y_m * d
    i_path = i_x - I_m * d
    return (pi_path=pi_path, y_path=y_path, i_path=i_path, d=d)
end

"""
Mirror of Caravello et al. (2025) cnfctl_pred_fn setup.

Inputs:
- Pi_m, Y_m, I_m: transmission maps (T x S)
- pi_x, y_x, i_x: baseline paths (T)
- i_hist_last: last observed policy rate before counterfactual start
- W_pi, W_y, W_i: T x T weight matrices
- lambda_di: wedge scale for initial-rate anchor

Returns named tuple:
(pi_path, y_path, i_path, d)
"""
function cnfctl_caravello(Pi_m, Y_m, I_m,
                          pi_x, y_x, i_x,
                          i_hist_last::Float64,
                          W_pi, W_y, W_i,
                          lambda_di::Float64)
    T = size(Pi_m, 1)
    i_init = vcat([i_hist_last], zeros(T - 1))
    wedge = lambda_di * (I_m' * i_init)

    sys = _caravello_system(Pi_m, Y_m, I_m, W_pi, W_y, W_i)
    return _caravello_solve(sys, Pi_m, Y_m, I_m, pi_x, y_x, i_x, wedge)
end

"""
Scenario/flexible style one-shot solve:
- build transmission maps over the forecast horizon
- build baseline forecast paths once
- call cnfctl_caravello once
"""
function cnfctl_caravello_flexible(Y, A_list, c,
                                   narr_point, hfi_point,
                                   fcst_date_idx::Int, fcst_hor::Int,
                                   i_hist_last::Float64,
                                   W_pi, W_y, W_i,
                                   lambda_di::Float64)
    T_hor = fcst_hor
    Pi_m = build_two_shock_map(narr_point[:, irf_cpi_idx], hfi_point[:, irf_cpi_idx], T_hor)
    Y_m = build_two_shock_map(narr_point[:, irf_gdp_idx], hfi_point[:, irf_gdp_idx], T_hor)
    I_m = build_two_shock_map(narr_point[:, irf_fr007_idx], hfi_point[:, irf_fr007_idx], T_hor)

    baseline = forecast_from_date(Y, A_list, c, fcst_date_idx - 1, T_hor)
    pi_x = baseline[:, cpi_idx]
    y_x = baseline[:, gdp_idx]
    i_x = baseline[:, fr007_idx]

    res = cnfctl_caravello(Pi_m, Y_m, I_m,
                           pi_x, y_x, i_x,
                           i_hist_last,
                           W_pi, W_y, W_i,
                           lambda_di)

    return (pi_cnfctl=res.pi_path, y_cnfctl=res.y_path, i_cnfctl=res.i_path, d=res.d,
            Pi_m=Pi_m, Y_m=Y_m, I_m=I_m)
end

function save_caravello_scenario_figure(sample::String, cf_start::Date,
                                        hist_dates, hist_data,
                                        actual_dates_window, actual_data,
                                        cnfctl_dates,
                                        pi_target, y_target,
                                        pi_base, y_base, i_base, e_base, ip_base,
                                        pi_car, y_car, i_car, e_car, ip_car,
                                        pi_flex, y_flex, i_flex, e_flex, ip_flex,
                                        out_dir)
    blue = RGB(0.45, 0.62, 0.70)
    green = RGB(0.20, 0.50, 0.20)

    fig = plot(layout=(1, 5), size=(2200, 420), margin=5Plots.mm,
        plot_title="Caravello Scenario from $(year(cf_start)) — sample $(sample)")

    all_plot_dates = vcat(hist_dates, actual_dates_window)
    all_plot_data = vcat(hist_data, actual_data)

    col_vars = [
        (idx=cpi_idx, base=pi_base, car=pi_car, flex=pi_flex, tgt=pi_target, ctitle="CPI YoY (%)"),
        (idx=gdp_idx, base=y_base, car=y_car, flex=y_flex, tgt=y_target, ctitle="Real GDP YoY (%)"),
        (idx=fr007_idx, base=i_base, car=i_car, flex=i_flex, tgt=nothing, ctitle="FR007 (%)"),
        (idx=neer_idx, base=e_base, car=e_car, flex=e_flex, tgt=nothing, ctitle="NEER YoY (%)"),
        (idx=ip_idx, base=ip_base, car=ip_car, flex=ip_flex, tgt=nothing, ctitle="IP YoY (%)"),
    ]

    for (sp, cv) in enumerate(col_vars)
        plot!(fig[sp], all_plot_dates, all_plot_data[:, cv.idx],
            color=:black, lw=2, label=(sp == 1 ? "Data" : ""))
        plot!(fig[sp], cnfctl_dates, cv.base,
            color=:gray, lw=1.5, ls=:dash, label=(sp == 1 ? "Forecast" : ""))
        if cv.tgt !== nothing
            plot!(fig[sp], cnfctl_dates, cv.tgt,
                color=:red, lw=1.5, ls=:dot, label=(sp == 1 ? "Target" : ""))
        end
        plot!(fig[sp], cnfctl_dates, cv.car,
            color=blue, lw=2.5, label=(sp == 1 ? "Caravello" : ""))
        plot!(fig[sp], cnfctl_dates, cv.flex,
            color=green, lw=2.0, ls=:dashdot, label=(sp == 1 ? "Flexible" : ""))
        hline!(fig[sp], [0], color=:gray, ls=:dot, alpha=0.3, label="")
        title!(fig[sp], cv.ctitle)
    end

    fpath = joinpath(out_dir, "cnfctl_scenario_caravello_$(year(cf_start))_s$(sample).png")
    savefig(fig, fpath)
    return fpath
end

function run_minimal_test()
    sample = "2022"
    cf_start = Date(2023, 1, 1)
    cf_end = Date(2025, 12, 1)
    rule_label = "CPI + GDP + NEER"

    println("="^70)
    println("Caravello-style counterfactual test")
    println("sample=$(sample), start=$(cf_start), end=$(cf_end), rule=$(rule_label)")
    println("="^70)

    inter_dir = intermediate_dir(sample)

    bvar = deserialize(joinpath(inter_dir, "bvar_results.jls"))
    narr_data = deserialize(joinpath(inter_dir, "narrative_irf_results.jls"))
    hfi_data = deserialize(joinpath(inter_dir, "hfi_irf_ratechange.jls"))

    A_list = bvar["A_list"]
    c_vec = bvar["c"]

    narr_point = narr_data["irf_point"]
    hfi_point = hfi_data["irf_point"]

    H_irf = min(narr_data["H"], hfi_data["H"])
    narr_point = narr_point[1:H_irf+1, :]
    hfi_point = hfi_point[1:H_irf+1, :]

    df = CSV.read(joinpath(DERIVED_DIR, "china_longterm_data.csv"), DataFrame)
    df.date = Date.(df.date)
    sort!(df, :date)
    syms = [:realgdp_monthly_yoy, :IP_yoy, :cpi, :FR007, :M2_growth, :neer_yoy, :US_IP_yoy]
    df = dropmissing(df, syms)

    dates_actual = df.date
    Y_actual = Matrix{Float64}(df[:, syms])

    fcst_date_idx = findfirst(d -> d >= cf_start, dates_actual)
    fcst_hor = length(cf_start:Month(1):cf_end)
    cnfctl_dates = [cf_start + Month(h - 1) for h in 1:fcst_hor]

    baseline_fc = forecast_from_date(Y_actual, A_list, c_vec, fcst_date_idx - 1, fcst_hor)
    pi_base = baseline_fc[:, cpi_idx]
    y_base = baseline_fc[:, gdp_idx]
    i_base = baseline_fc[:, fr007_idx]
    e_base = baseline_fc[:, neer_idx]
    ip_base = baseline_fc[:, ip_idx]

    T_actual_avail = min(fcst_hor, size(Y_actual, 1) - fcst_date_idx + 1)
    actual_data = Y_actual[fcst_date_idx:min(fcst_date_idx + T_actual_avail - 1, end), :]
    actual_dates_window = dates_actual[fcst_date_idx:min(fcst_date_idx + T_actual_avail - 1, end)]

    hist_start_idx = max(1, fcst_date_idx - 12)
    hist_dates = dates_actual[hist_start_idx:fcst_date_idx - 1]
    hist_data = Y_actual[hist_start_idx:fcst_date_idx - 1, :]

    i_hist_last = Y_actual[fcst_date_idx - 1, fr007_idx]

    W_pi = Matrix{Float64}(I, fcst_hor, fcst_hor)
    W_y = Matrix{Float64}(I, fcst_hor, fcst_hor)
    W_i = Matrix{Float64}(I, fcst_hor, fcst_hor)
    lambda_di = 1.0

    car = cnfctl_caravello_flexible(
        Y_actual, A_list, c_vec,
        narr_point, hfi_point,
        fcst_date_idx, fcst_hor,
        i_hist_last,
        W_pi, W_y, W_i,
        lambda_di,
    )

    # Existing cnfctl_flexible scenario output (saved previously by Counterfactual.jl)
    saved_path = joinpath(intermediate_dir("2025"), "counterfactual_results.jls")
    saved = deserialize(saved_path)
    key = "scenario_$(sample)_$(year(cf_start))_$(rule_label)"

    println("\nLoaded saved comparison key: ", key)
    if haskey(saved["results"], key)
        existing_res = saved["results"][key]["res"]

        E_m = build_two_shock_map(narr_point[:, irf_neer_idx], hfi_point[:, irf_neer_idx], fcst_hor)
        IP_m = build_two_shock_map(narr_point[:, irf_ip_idx], hfi_point[:, irf_ip_idx], fcst_hor)

        # Caravello sign convention: paths are baseline - M*d
        e_car = e_base - E_m * car.d
        ip_car = ip_base - IP_m * car.d

        # Existing flexible output stores NEER as deviation; IP already level.
        e_flex = e_base .+ existing_res.e_cnfctl
        ip_flex = existing_res.ip_cnfctl

        println("\npi_path_caravello = ", round.(car.pi_cnfctl, digits=4))
        println("pi_path_flexible  = ", round.(existing_res.pi_cnfctl, digits=4))

        println("\ny_path_caravello  = ", round.(car.y_cnfctl, digits=4))
        println("y_path_flexible   = ", round.(existing_res.y_cnfctl, digits=4))

        println("\ni_path_caravello  = ", round.(car.i_cnfctl, digits=4))
        println("i_path_flexible   = ", round.(existing_res.i_cnfctl, digits=4))

        println("\nFirst 12 months side-by-side")
        for t in 1:min(12, fcst_hor)
            @printf("%02d | pi car=%8.4f flex=%8.4f | y car=%8.4f flex=%8.4f | i car=%8.4f flex=%8.4f\n",
                    t,
                    car.pi_cnfctl[t], existing_res.pi_cnfctl[t],
                    car.y_cnfctl[t], existing_res.y_cnfctl[t],
                    car.i_cnfctl[t], existing_res.i_cnfctl[t])
        end

        pi_target = [cpi_target_for_year(year(d)) for d in cnfctl_dates]
        y_target = [gdp_target_for_year(year(d)) for d in cnfctl_dates]

        main_dir = main_results_dir(sample)
        fig_path = save_caravello_scenario_figure(
            sample, cf_start,
            hist_dates, hist_data,
            actual_dates_window, actual_data,
            cnfctl_dates,
            pi_target, y_target,
            pi_base, y_base, i_base, e_base, ip_base,
            car.pi_cnfctl, car.y_cnfctl, car.i_cnfctl, e_car, ip_car,
            existing_res.pi_cnfctl, existing_res.y_cnfctl, existing_res.i_cnfctl, e_flex, ip_flex,
            main_dir,
        )
        println("\nSaved figure: ", fig_path)
    else
        println("Saved key not found in ", saved_path)
        println("Available keys count: ", length(keys(saved["results"])))
    end

    return car
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_minimal_test()
end
