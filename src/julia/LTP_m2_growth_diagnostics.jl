##
using CSV, DataFrames
using Statistics
using Plots
using Dates
using Printf
using LinearAlgebra
using Serialization

include(joinpath(@__DIR__, "common.jl"))

# Compare the LTP BVAR under two M2 transformations:
#   1. m2_yoy_pct:       (M2 / lag(M2, 12) - 1) * 100
#   2. m2_yoy_leveldiff: M2 - lag(M2, 12)
#
# The script does not modify derived data. It reconstructs both M2 variants
# from the cached M2 level series and swaps only M2_growth before estimating.

const VAR_SYMS = [:realgdp_monthly_yoy, :IP_yoy, :cpi, :FR007, :M2_growth, :neer_yoy, :US_IP_yoy]
const VAR_LABELS = ["Real GDP", "IP", "CPI", "FR007", "M2", "NEER", "US IP"]
const P_LAGS = 6
const H_WOLD = 120
const H_FC = 24
const CF_START = Date(2023, 1, 1)
const CF_END = Date(2025, 12, 1)

function lag_vec(x::AbstractVector, k::Int)
    out = Vector{Union{Missing, eltype(x)}}(missing, length(x))
    for i in k+1:length(x)
        out[i] = x[i-k]
    end
    return out
end

function max_companion_eigenvalue(A_list::Vector{Matrix{Float64}})
    p_loc = length(A_list)
    n_loc = size(A_list[1], 1)
    F = zeros(n_loc * p_loc, n_loc * p_loc)
    for L in 1:p_loc
        F[1:n_loc, (L-1)*n_loc+1 : L*n_loc] = A_list[L]
    end
    if p_loc > 1
        F[n_loc+1:end, 1:n_loc*(p_loc-1)] = Matrix(I, n_loc*(p_loc-1), n_loc*(p_loc-1))
    end
    return maximum(abs.(eigvals(F)))
end

function compute_wold_ltp(A_list::Vector{Matrix{Float64}}, H::Int)
    p_loc = length(A_list)
    n_loc = size(A_list[1], 1)
    Psi = Matrix{Float64}[Matrix(I, n_loc, n_loc)]
    for h in 1:H
        Psi_h = zeros(n_loc, n_loc)
        for j in 1:min(h, p_loc)
            Psi_h .+= A_list[j] * Psi[h - j + 1]
        end
        push!(Psi, Psi_h)
    end
    return Psi
end

function compute_residuals_ltp_diag(Y::Matrix{Float64}, A_list, c::Vector{Float64})
    T_loc, n_loc = size(Y)
    p_loc = length(A_list)
    resids = zeros(T_loc - p_loc, n_loc)
    for t in p_loc+1:T_loc
        y_hat = copy(c)
        for j in 1:p_loc
            y_hat .+= A_list[j] * Y[t - j, :]
        end
        resids[t - p_loc, :] = Y[t, :] - y_hat
    end
    return resids
end

function forecast_from_diag(Y::Matrix{Float64}, A_list, c::Vector{Float64},
                            start_idx::Int, H_fc::Int)
    p_loc = length(A_list)
    history = [Y[start_idx - j + 1, :] for j in 1:p_loc]
    fc = zeros(H_fc, size(Y, 2))
    for h in 1:H_fc
        y_hat = copy(c)
        for j in 1:p_loc
            y_hat .+= A_list[j] * history[j]
        end
        fc[h, :] = y_hat
        pushfirst!(history, y_hat)
        pop!(history)
    end
    return fc
end

function add_m2_variants(df::DataFrame)
    m2_path = joinpath(PROJECT_ROOT, "data", "raw", "cache", "m2_supply.csv")
    m2 = CSV.read(m2_path, DataFrame)
    m2.date = Date.(m2.date)
    sort!(m2, :date)
    m2.M2_l12 = lag_vec(m2.M2, 12)
    m2.m2_yoy_pct = (m2.M2 ./ m2.M2_l12 .- 1.0) .* 100.0
    m2.m2_yoy_leveldiff = m2.M2 .- m2.M2_l12

    out = leftjoin(df, m2[:, [:date, :M2, :m2_yoy_pct, :m2_yoy_leveldiff]], on=:date)
    return out
end

function fit_ltp_variant(df_base::DataFrame, m2_col::Symbol, sample)
    df = copy(df_base)
    df.M2_growth = df[:, m2_col]

    df_est = dropmissing(df, VAR_SYMS)
    sort!(df_est, :date)
    df_est = filter(r -> r.date >= sample.start_date && r.date <= sample.end_date, df_est)

    T_full = nrow(df_est)
    Y_full = Matrix{Float64}(df_est[:, VAR_SYMS])
    dates_full = df_est.date
    n = length(VAR_SYMS)
    p = P_LAGS

    covid_dates_all = [Date(2020,1,1), Date(2020,2,1), Date(2020,3,1), Date(2020,4,1)]
    covid_dates = filter(d -> d >= sample.start_date && d <= sample.end_date, covid_dates_all)
    n_covid = length(covid_dates)

    Teff = T_full - p
    Y_dep = Y_full[p+1:end, :]
    dates_est = dates_full[p+1:end]
    k = 1 + n * p + n_covid

    X = zeros(Teff, k)
    X[:, 1] .= 1.0
    for L in 1:p
        X[:, 1 + (L-1)*n+1 : 1 + L*n] = Y_full[p+1-L:end-L, :]
    end
    for (ci, cd) in enumerate(covid_dates)
        col = 1 + n*p + ci
        for t in 1:Teff
            X[t, col] = (dates_est[t] == cd) ? 1.0 : 0.0
        end
    end

    B_ols = X \ Y_dep
    U_ols = Y_dep - X * B_ols
    sigma2_ols = vec(var(U_ols, dims=1))

    lambda = 0.2
    d = 1.0

    B0 = zeros(k, n)
    for j in 1:n
        B0[1 + j, j] = 1.0
    end

    Omega_diag = zeros(k, n)
    for j in 1:n
        Omega_diag[1, j] = 100.0 * sigma2_ols[j]
        for L in 1:p
            for i in 1:n
                row = 1 + (L-1)*n + i
                Omega_diag[row, j] = (lambda / L^d)^2 * (sigma2_ols[j] / sigma2_ols[i])
            end
        end
        for ci in 1:n_covid
            Omega_diag[1 + n*p + ci, j] = 100.0 * sigma2_ols[j]
        end
    end

    B_post = zeros(k, n)
    XtX = X' * X
    for j in 1:n
        Omega_j_inv = Diagonal(1.0 ./ Omega_diag[:, j])
        V_j_inv = Omega_j_inv + XtX
        V_j = Symmetric(inv(V_j_inv))
        b_j = V_j * (Omega_j_inv * B0[:, j] + X' * Y_dep[:, j])
        B_post[:, j] = b_j
    end

    U_post = Y_dep - X * B_post
    Sigma_u = (U_post' * U_post) / Teff
    A_list = get_lag_matrices(B_post, n, p)
    c_vec = B_post[1, :]
    Psi = compute_wold_ltp(A_list, H_WOLD)
    residuals = compute_residuals_ltp_diag(Y_full, A_list, c_vec)
    fc = forecast_from_diag(Y_full, A_list, c_vec, T_full, H_FC)
    fc_dates = [dates_full[end] + Month(h) for h in 1:H_FC]

    return (
        sample=sample,
        df_est=df_est,
        dates_est=dates_est,
        Y_dep=Y_dep,
        X=X,
        B_post=B_post,
        Sigma_u=Sigma_u,
        A_list=A_list,
        Psi=Psi,
        residuals=residuals,
        forecast=fc,
        forecast_dates=fc_dates,
        max_eig=max_companion_eigenvalue(A_list),
        sigma2_ols=sigma2_ols,
    )
end

function forecast_frame(sample_label::String, variant::String, result)
    rows = DataFrame()
    for j in eachindex(VAR_SYMS)
        append!(rows, DataFrame(
            sample=sample_label,
            variant=variant,
            date=result.forecast_dates,
            variable=String(VAR_SYMS[j]),
            forecast=result.forecast[:, j],
        ))
    end
    return rows
end

function counterfactual_baseline_from_result(df_base::DataFrame, m2_col::Symbol, result)
    df = copy(df_base)
    df.M2_growth = df[:, m2_col]
    df_est = dropmissing(df, VAR_SYMS)
    sort!(df_est, :date)
    Y_actual = Matrix{Float64}(df_est[:, VAR_SYMS])
    cf_start_idx = findfirst(d -> d >= CF_START, df_est.date)
    cf_hor = length(CF_START:Month(1):CF_END)
    fc = forecast_from_diag(Y_actual, result.A_list, result.B_post[1, :], cf_start_idx - 1, cf_hor)
    return DataFrame(
        date=[CF_START + Month(h - 1) for h in 1:cf_hor],
        gdp_forecast=fc[:, 1],
    )
end

function forecast_saved_bvar_on_df(df_base::DataFrame, sample_label::String)
    bvar_path = joinpath(intermediate_dir(sample_label), "bvar_results.jls")
    if !isfile(bvar_path)
        return nothing
    end
    bvar = deserialize(bvar_path)
    df_est = dropmissing(df_base, VAR_SYMS)
    sort!(df_est, :date)
    Y_actual = Matrix{Float64}(df_est[:, VAR_SYMS])
    cf_start_idx = findfirst(d -> d >= CF_START, df_est.date)
    cf_hor = length(CF_START:Month(1):CF_END)
    fc = forecast_from_diag(Y_actual, bvar["A_list"], bvar["c"], cf_start_idx - 1, cf_hor)
    return (
        path=bvar_path,
        mtime=stat(bvar_path).mtime,
        data_mtime=stat(joinpath(DERIVED_DIR, "china_longterm_data.csv")).mtime,
        frame=DataFrame(
            date=[CF_START + Month(h - 1) for h in 1:cf_hor],
            saved_bvar_gdp_forecast=fc[:, 1],
        ),
    )
end

function head_longterm_data()
    try
        raw = read(`git show HEAD:Data/derived/china_longterm_data.csv`, String)
        df = CSV.read(IOBuffer(raw), DataFrame)
        df.date = Date.(df.date)
        sort!(df, :date)
        return df
    catch err
        @warn "Could not read HEAD:Data/derived/china_longterm_data.csv" err
        return nothing
    end
end

function summary_frame(sample_label::String, variant::String, result)
    rows = DataFrame(
        sample=String[],
        variant=String[],
        variable=String[],
        ols_resid_std=Float64[],
        post_resid_std=Float64[],
        sigma_u_diag=Float64[],
        max_companion_eigenvalue=Float64[],
    )
    for j in eachindex(VAR_SYMS)
        push!(rows, (
            sample_label,
            variant,
            String(VAR_SYMS[j]),
            sqrt(result.sigma2_ols[j]),
            std(result.residuals[:, j]),
            result.Sigma_u[j, j],
            result.max_eig,
        ))
    end
    return rows
end

function save_comparison_plots(out_dir::String, sample_label::String, pct_res, old_res, fc_comp::DataFrame)
    gdp = filter(r -> r.variable == "realgdp_monthly_yoy", fc_comp)
    p_gdp = plot(title="LTP Forecast Comparison: GDP YoY ($(sample_label))",
        xlabel="Date", ylabel="% YoY")
    for variant in unique(gdp.variant)
        tmp = filter(r -> r.variant == variant, gdp)
        plot!(p_gdp, tmp.date, tmp.forecast, label=variant, linewidth=2, marker=:circle)
        start_date = tmp.date[1]
        start_value = tmp.forecast[1]
        annotate!(p_gdp, start_date, start_value,
            text(@sprintf("  %s: %.2f", variant, start_value), 8, :left))
    end
    savefig(p_gdp, joinpath(out_dir, "gdp_forecast_pct_vs_leveldiff_$(sample_label).png"))

    diff_df = unstack(fc_comp, [:sample, :date, :variable], :variant, :forecast)
    if all(x -> x in names(diff_df), ["m2_yoy_pct", "m2_yoy_leveldiff"])
        diff_df.diff_pct_minus_leveldiff = diff_df.m2_yoy_pct .- diff_df.m2_yoy_leveldiff
        p_diff = plot(title="Forecast Difference: pct M2 minus level-diff M2 ($(sample_label))",
            xlabel="Date", ylabel="Forecast difference")
        for variable in unique(diff_df.variable)
            tmp = filter(r -> r.variable == variable, diff_df)
            plot!(p_diff, tmp.date, tmp.diff_pct_minus_leveldiff, label=variable, linewidth=1.5)
        end
        hline!(p_diff, [0], color=:black, linestyle=:dash, alpha=0.4, label="")
        savefig(p_diff, joinpath(out_dir, "forecast_differences_$(sample_label).png"))
    end

    wold_pct = [pct_res.Psi[h+1][1, 1] for h in 0:H_WOLD]
    wold_old = [old_res.Psi[h+1][1, 1] for h in 0:H_WOLD]
    p_wold = plot(0:H_WOLD, wold_pct, label="m2_yoy_pct", linewidth=2,
        xlabel="Horizon (months)", ylabel="Psi_h[GDP,GDP]",
        title="GDP Wold Diagonal Comparison ($(sample_label))")
    plot!(p_wold, 0:H_WOLD, wold_old, label="m2_yoy_leveldiff", linewidth=2)
    hline!(p_wold, [0], color=:black, linestyle=:dash, alpha=0.4, label="")
    savefig(p_wold, joinpath(out_dir, "gdp_wold_diagonal_$(sample_label).png"))
end

function save_counterfactual_baseline_check(out_dir::String, sample_label::String,
                                            df::DataFrame, pct_res, old_res)
    pct_cf = rename(counterfactual_baseline_from_result(df, :m2_yoy_pct, pct_res),
        :gdp_forecast => :fresh_m2_yoy_pct)
    old_cf = rename(counterfactual_baseline_from_result(df, :m2_yoy_leveldiff, old_res),
        :gdp_forecast => :fresh_m2_yoy_leveldiff)
    comp = innerjoin(pct_cf, old_cf, on=:date)

    saved_current = forecast_saved_bvar_on_df(df, sample_label)
    if saved_current !== nothing
        saved_current_frame = rename(saved_current.frame,
            :saved_bvar_gdp_forecast => :saved_bvar_current_data_mismatch)
        comp = leftjoin(comp, saved_current_frame, on=:date)
        comp.saved_current_minus_fresh_pct = comp.saved_bvar_current_data_mismatch .- comp.fresh_m2_yoy_pct
    end

    head_df = head_longterm_data()
    saved_head = head_df === nothing ? nothing : forecast_saved_bvar_on_df(head_df, sample_label)
    if saved_head !== nothing
        saved_head_frame = rename(saved_head.frame,
            :saved_bvar_gdp_forecast => :saved_bvar_head_data)
        comp = leftjoin(comp, saved_head_frame, on=:date)
        comp.saved_head_minus_fresh_leveldiff = comp.saved_bvar_head_data .- comp.fresh_m2_yoy_leveldiff
    end

    comp.sample .= sample_label
    select!(comp, :sample, :)
    out_csv = joinpath(out_dir, "counterfactual_baseline_gdp_check_$(sample_label).csv")
    CSV.write(out_csv, comp)

    p = plot(comp.date, comp.fresh_m2_yoy_pct, label="fresh m2_yoy_pct", linewidth=2,
        xlabel="Date", ylabel="GDP YoY forecast",
        title="Counterfactual Baseline GDP Check ($(sample_label))")
    plot!(p, comp.date, comp.fresh_m2_yoy_leveldiff, label="fresh m2_yoy_leveldiff", linewidth=2)
    if "saved_bvar_head_data" in names(comp)
        plot!(p, comp.date, comp.saved_bvar_head_data,
            label="saved BVAR + HEAD data", linewidth=2, linestyle=:dash)
    end
    if "saved_bvar_current_data_mismatch" in names(comp)
        plot!(p, comp.date, comp.saved_bvar_current_data_mismatch,
            label="saved BVAR + current data mismatch", linewidth=1.5, linestyle=:dot)
    end
    hline!(p, [0], color=:black, linestyle=:dash, alpha=0.4, label="")
    savefig(p, joinpath(out_dir, "counterfactual_baseline_gdp_check_$(sample_label).png"))

    if saved_head !== nothing
        println(@sprintf("  saved BVAR + HEAD data GDP: first=%.2f  min=%.2f  max=%.2f",
            comp.saved_bvar_head_data[1],
            minimum(skipmissing(comp.saved_bvar_head_data)),
            maximum(skipmissing(comp.saved_bvar_head_data))))
    end
    if saved_current !== nothing
        is_stale = saved_current.mtime < saved_current.data_mtime
        println(@sprintf("  saved BVAR + current data mismatch GDP: first=%.2f  min=%.2f  max=%.2f  stale=%s",
            comp.saved_bvar_current_data_mismatch[1],
            minimum(skipmissing(comp.saved_bvar_current_data_mismatch)),
            maximum(skipmissing(comp.saved_bvar_current_data_mismatch)),
            string(is_stale)))
    end
    println(@sprintf("  fresh pct baseline GDP: first=%.2f  min=%.2f  max=%.2f",
        comp.fresh_m2_yoy_pct[1], minimum(comp.fresh_m2_yoy_pct), maximum(comp.fresh_m2_yoy_pct)))
    println("  saved counterfactual baseline check: ", out_csv)
end

function main()
    out_dir = joinpath(PROJECT_ROOT, "outputs", "diagnostics", "ltp_m2_growth_comparison")
    mkpath(out_dir)

    df_raw = CSV.read(joinpath(DERIVED_DIR, "china_longterm_data.csv"), DataFrame)
    df_raw.date = Date.(df_raw.date)
    sort!(df_raw, :date)
    df = add_m2_variants(df_raw)

    all_summary = DataFrame()
    all_forecasts = DataFrame()

    println("\n", "="^70)
    println("  LTP M2 growth diagnostics")
    println("="^70)
    println("Data: ", joinpath(DERIVED_DIR, "china_longterm_data.csv"))
    println("M2 levels: ", joinpath(PROJECT_ROOT, "data", "raw", "cache", "m2_supply.csv"))
    println("Output: ", out_dir)

    for s in SAMPLES
        println("\nSample $(s.label): $(s.start_date) to $(s.end_date)")
        pct_res = fit_ltp_variant(df, :m2_yoy_pct, s)
        old_res = fit_ltp_variant(df, :m2_yoy_leveldiff, s)

        append!(all_summary, summary_frame(s.label, "m2_yoy_pct", pct_res))
        append!(all_summary, summary_frame(s.label, "m2_yoy_leveldiff", old_res))

        fc_comp = vcat(
            forecast_frame(s.label, "m2_yoy_pct", pct_res),
            forecast_frame(s.label, "m2_yoy_leveldiff", old_res),
        )
        append!(all_forecasts, fc_comp)
        save_comparison_plots(out_dir, s.label, pct_res, old_res, fc_comp)
        save_counterfactual_baseline_check(out_dir, s.label, df, pct_res, old_res)

        gdp_pct = filter(r -> r.variant == "m2_yoy_pct" && r.variable == "realgdp_monthly_yoy", fc_comp)
        gdp_old = filter(r -> r.variant == "m2_yoy_leveldiff" && r.variable == "realgdp_monthly_yoy", fc_comp)
        gdp_diff = gdp_pct.forecast .- gdp_old.forecast
        println(@sprintf("  max eig: pct=%.4f  leveldiff=%.4f", pct_res.max_eig, old_res.max_eig))
        println(@sprintf("  GDP forecast diff pct-leveldiff: mean=%.4f  maxabs=%.4f",
            mean(gdp_diff), maximum(abs.(gdp_diff))))
    end

    fc_wide = unstack(all_forecasts, [:sample, :date, :variable], :variant, :forecast)
    if all(x -> x in names(fc_wide), ["m2_yoy_pct", "m2_yoy_leveldiff"])
        fc_wide.diff_pct_minus_leveldiff = fc_wide.m2_yoy_pct .- fc_wide.m2_yoy_leveldiff
    end

    CSV.write(joinpath(out_dir, "ltp_m2_variant_summary.csv"), all_summary)
    CSV.write(joinpath(out_dir, "ltp_m2_forecasts_long.csv"), all_forecasts)
    CSV.write(joinpath(out_dir, "ltp_m2_forecast_comparison.csv"), fc_wide)

    println("\nSaved:")
    println("  ", joinpath(out_dir, "ltp_m2_variant_summary.csv"))
    println("  ", joinpath(out_dir, "ltp_m2_forecasts_long.csv"))
    println("  ", joinpath(out_dir, "ltp_m2_forecast_comparison.csv"))
end

main()
