##
# =============================================================================
# HFI ZERO-DRAW DIAGNOSTICS
# =============================================================================
#
# Usage:
#   julia src/julia/HFIShocks_zero_draw_diagnostics.jl
#
# Optional sample labels:
#   julia src/julia/HFIShocks_zero_draw_diagnostics.jl 2025 2022
#
# Optional draw count:
#   julia src/julia/HFIShocks_zero_draw_diagnostics.jl --draws=1000 2025
#
# This script tests the proposed adjustment to HFIShocks.jl's posterior-draw
# handling: when no valid posterior draws survive filtering, do not keep the
# existing all-zero placeholder draw. The script leaves production outputs
# untouched and prints what current vs adjusted behavior would return.
# =============================================================================

using CSV, DataFrames
using Dates
using LinearAlgebra
using Plots
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "common.jl"))

const HFI_VAR_SYMS = [:realgdp_monthly_yoy, :cpi, :FR007, :neer_yoy, :IP_yoy]
const HFI_VAR_LABELS = ["Real GDP Growth", "CPI", "FR007", "Real Effective Exchange Rate", "Industrial Value Added"]
const HFI_INSTRUMENTS = [:shock_policy, :shock_policy_change]
const HFI_P = 6
const HFI_H = 24

function fmt(x; digits::Int=4)
    if x === nothing || ismissing(x) || !isfinite(Float64(x))
        return "NA"
    end
    return @sprintf("%.*f", digits, Float64(x))
end

function build_hfi_posterior(df::DataFrame, instrument_sym::Symbol; λ::Float64=0.2, d::Float64=1.0)
    all_syms = vcat(HFI_VAR_SYMS, [instrument_sym])
    df_bvar = dropmissing(df, all_syms)
    if nrow(df_bvar) <= HFI_P + 5
        error("Not enough complete BVAR rows for $(instrument_sym): $(nrow(df_bvar))")
    end

    n = length(HFI_VAR_SYMS)
    Y_full = Matrix{Float64}(df_bvar[:, HFI_VAR_SYMS])
    z_full = Float64.(df_bvar[:, instrument_sym])
    T_raw = size(Y_full, 1)
    Teff = T_raw - HFI_P
    Y_dep = Y_full[HFI_P+1:end, :]
    X_var = ones(Teff, 1 + n * HFI_P)
    for L in 1:HFI_P
        X_var[:, 1 + (L-1)*n+1 : 1 + L*n] = Y_full[HFI_P+1-L:end-L, :]
    end
    z = z_full[HFI_P+1:end]
    dates_est = df_bvar.date[HFI_P+1:end]
    k = size(X_var, 2)

    B_ols = X_var \ Y_dep
    U_ols = Y_dep - X_var * B_ols
    sigma2_ols = vec(var(U_ols, dims=1))

    B0 = zeros(k, n)
    for j in 1:n
        B0[1 + j, j] = 1.0
    end

    Omega_diag = zeros(k, n)
    for j in 1:n
        Omega_diag[1, j] = 100.0 * sigma2_ols[j]
        for L in 1:HFI_P, i in 1:n
            row = 1 + (L-1)*n + i
            Omega_diag[row, j] = (λ / L^d)^2 * (sigma2_ols[j] / sigma2_ols[i])
        end
    end

    B_post = zeros(k, n)
    V_post = Vector{Matrix{Float64}}(undef, n)
    for j in 1:n
        Omega_j_inv = Diagonal(1.0 ./ Omega_diag[:, j])
        V_j_post_inv = Omega_j_inv + X_var' * X_var
        V_j_post = Symmetric(inv(V_j_post_inv))
        B_post[:, j] = V_j_post * (Omega_j_inv * B0[:, j] + X_var' * Y_dep[:, j])
        V_post[j] = Matrix(V_j_post)
    end

    U_post = Y_dep - X_var * B_post
    policy_col = findfirst(==(:FR007), HFI_VAR_SYMS)
    gdp_col = findfirst(==(:realgdp_monthly_yoy), HFI_VAR_SYMS)
    b_point, F_stat, gamma_hat = iv_identify(U_post, z, policy_col, gdp_col)

    return (
        B_post=B_post,
        V_post=V_post,
        Y_dep=Y_dep,
        X_var=X_var,
        z=z,
        dates_est=dates_est,
        policy_col=policy_col,
        gdp_col=gdp_col,
        b_point=b_point,
        F_stat=F_stat,
        gamma_hat=gamma_hat,
    )
end

function current_finalize_draws(draws::Array{Float64, 3}, valid_draws::Int)
    return draws[1:max(valid_draws, 1), :, :]
end

function adjusted_finalize_draws(draws::Array{Float64, 3}, valid_draws::Int; mode::Symbol=:error)
    if valid_draws > 0
        return draws[1:valid_draws, :, :]
    end
    if mode == :empty
        return draws[1:0, :, :]
    elseif mode == :error
        error("No valid posterior draws survived filtering; refusing to serialize a dummy all-zero draw.")
    else
        error("Unknown zero-draw mode $(mode). Use :empty or :error.")
    end
end

function irf_bands(draws::Array{Float64, 3})
    n_draws, H_len, n = size(draws)
    if n_draws == 0
        error("Cannot compute IRF bands from an empty posterior draw array.")
    end

    median = zeros(H_len, n)
    lo68 = zeros(H_len, n)
    hi68 = zeros(H_len, n)
    lo90 = zeros(H_len, n)
    hi90 = zeros(H_len, n)
    for h in 1:H_len, j in 1:n
        x = draws[:, h, j]
        median[h, j] = quantile(x, 0.50)
        lo68[h, j] = quantile(x, 0.16)
        hi68[h, j] = quantile(x, 0.84)
        lo90[h, j] = quantile(x, 0.05)
        hi90[h, j] = quantile(x, 0.95)
    end
    return (median=median, lo68=lo68, hi68=hi68, lo90=lo90, hi90=hi90)
end

function save_irf_plot(draws::Array{Float64, 3}, irf_point::Union{Nothing, Matrix{Float64}},
                       sample_label::String, instrument::Symbol, setting::String)
    bands = irf_bands(draws)
    n = length(HFI_VAR_SYMS)
    p_plots = []
    for j in 1:n
        plt = plot(0:HFI_H, bands.median[:, j],
            title=HFI_VAR_LABELS[j],
            label="Median",
            xlabel="Months",
            ylabel="Response",
            legend=:best,
            linewidth=2.5,
            color=:darkred)
        plot!(plt, 0:HFI_H, bands.lo90[:, j],
            fillrange=bands.hi90[:, j],
            fillalpha=0.18,
            fillcolor=:pink,
            linealpha=0,
            label="90% CS")
        plot!(plt, 0:HFI_H, bands.lo68[:, j],
            fillrange=bands.hi68[:, j],
            fillalpha=0.35,
            fillcolor=:salmon,
            linealpha=0,
            label="68% CS")
        if irf_point !== nothing
            plot!(plt, 0:HFI_H, irf_point[:, j],
                label="Point",
                color=:black,
                linestyle=:dash,
                linewidth=1.4)
        end
        hline!([0], color=:gray, linestyle=:dash, alpha=0.5, label="")
        push!(p_plots, plt)
    end

    n_cols = 3
    n_rows = ceil(Int, n / n_cols)
    fig = plot(p_plots...,
        layout=(n_rows, n_cols),
        size=(380 * n_cols, 300 * n_rows),
        plot_title="HFI zero-draw diagnostics: $(instrument), $(setting) ($(sample_label))")

    out_dir = diagnostics_dir(sample_label)
    out_path = joinpath(out_dir, "hfi_irf_$(instrument)_$(setting).png")
    savefig(fig, out_path)
    println("    saved plot:      ", out_path)
    return out_path
end

function simulate_draws(post; n_draws::Int=500, irf_threshold::Float64=500.0)
    n = length(HFI_VAR_SYMS)
    k = size(post.X_var, 2)
    irf_draws = zeros(n_draws, HFI_H + 1, n)
    n_valid = 0
    n_iv_fail = 0
    n_huge_irf = 0

    Random.seed!(42)
    for _ in 1:n_draws
        B_draw = zeros(k, n)
        for j in 1:n
            C_cov = cholesky(Symmetric(post.V_post[j] + 1e-12 * I(k)); check=false)
            if !issuccess(C_cov)
                E = eigen(Symmetric(post.V_post[j]))
                sqrtV = E.vectors * Diagonal(sqrt.(max.(E.values, 1e-12)))
                B_draw[:, j] = post.B_post[:, j] + sqrtV * randn(k)
            else
                B_draw[:, j] = post.B_post[:, j] + C_cov.L * randn(k)
            end
        end

        U_draw = post.Y_dep - post.X_var * B_draw
        b_draw, _, _ = iv_identify(U_draw, post.z, post.policy_col, post.gdp_col)
        if isnothing(b_draw)
            n_iv_fail += 1
            continue
        end

        A_draw = get_lag_matrices(B_draw, n, HFI_P)
        C_draw = compute_ma(A_draw, n, HFI_P, HFI_H)
        irf_draw = compute_irfs(C_draw, b_draw, HFI_H)
        if any(abs.(irf_draw) .> irf_threshold)
            n_huge_irf += 1
            continue
        end

        n_valid += 1
        irf_draws[n_valid, :, :] = irf_draw
    end

    return (irf_draws=irf_draws, n_valid=n_valid, n_iv_fail=n_iv_fail,
            n_huge_irf=n_huge_irf, n_draws=n_draws)
end

function print_zero_draw_comparison(draws_result)
    current_draws = current_finalize_draws(draws_result.irf_draws, draws_result.n_valid)
    adjusted_empty = adjusted_finalize_draws(draws_result.irf_draws, draws_result.n_valid; mode=:empty)

    println("  current shape:        ", size(current_draws))
    println("  adjusted empty shape: ", size(adjusted_empty))
    println("  current all zero:     ", all(current_draws .== 0.0))

    try
        adjusted_finalize_draws(draws_result.irf_draws, draws_result.n_valid; mode=:error)
        println("  adjusted error mode:  no error")
    catch err
        println("  adjusted error mode:  ", err.msg)
    end
end

function run_for_sample(df_raw::DataFrame, s; n_draws::Int=500)
    println("\n", "="^78)
    println("HFI zero-draw diagnostics: $(s.start_date) to $(s.end_date) [$(s.label)]")
    println("="^78)

    df = filter(r -> !ismissing(r.date) && r.date >= s.start_date && r.date <= s.end_date, df_raw)
    println("Rows in date range: ", nrow(df))

    for instrument in HFI_INSTRUMENTS
        println("\nInstrument: ", instrument)
        post = build_hfi_posterior(df, instrument)
        println("  effective sample: ", first(post.dates_est), " to ", last(post.dates_est),
            " (T=", length(post.z), ")")
        println("  point IV status:  ", isnothing(post.b_point) ? "failed" : "identified")
        println("  first-stage F:    ", fmt(post.F_stat, digits=3))
        println("  corr(z,u_policy): ", fmt(cor(post.z, post.Y_dep[:, post.policy_col] - post.X_var * post.B_post[:, post.policy_col]), digits=6))

        if isnothing(post.b_point)
            println("  Skipping posterior draw simulation because point IV failed.")
            continue
        end

        A_post = get_lag_matrices(post.B_post, length(HFI_VAR_SYMS), HFI_P)
        C_post = compute_ma(A_post, length(HFI_VAR_SYMS), HFI_P, HFI_H)
        irf_point = compute_irfs(C_post, post.b_point, HFI_H)

        println("  Actual filter run with n_draws=", n_draws)
        actual = simulate_draws(post; n_draws=n_draws)
        println("    valid draws:     ", actual.n_valid, " / ", actual.n_draws)
        println("    IV failures:     ", actual.n_iv_fail)
        println("    huge IRF skips:  ", actual.n_huge_irf)
        print_zero_draw_comparison(actual)

        current_actual = current_finalize_draws(actual.irf_draws, actual.n_valid)
        adjusted_actual = adjusted_finalize_draws(actual.irf_draws, actual.n_valid; mode=:empty)
        save_irf_plot(current_actual, irf_point, s.label, instrument, "current_actual")
        save_irf_plot(adjusted_actual, irf_point, s.label, instrument, "adjusted_actual")

        println("  Forced zero-valid run (threshold=0.0) to test the adjustment path")
        forced = simulate_draws(post; n_draws=min(n_draws, 50), irf_threshold=0.0)
        println("    valid draws:     ", forced.n_valid, " / ", forced.n_draws)
        println("    IV failures:     ", forced.n_iv_fail)
        println("    huge IRF skips:  ", forced.n_huge_irf)
        print_zero_draw_comparison(forced)

        current_forced = current_finalize_draws(forced.irf_draws, forced.n_valid)
        save_irf_plot(current_forced, irf_point, s.label, instrument, "current_forced_zero")
        adjusted_forced = adjusted_finalize_draws(forced.irf_draws, forced.n_valid; mode=:empty)
        if size(adjusted_forced, 1) == 0
            println("    adjusted forced: no IRF plot; zero valid draws are exposed as empty output")
        else
            save_irf_plot(adjusted_forced, irf_point, s.label, instrument, "adjusted_forced_zero")
        end
    end
end

function main()
    n_draws = 500
    labels = String[]
    for arg in ARGS
        if startswith(arg, "--draws=")
            n_draws = parse(Int, replace(arg, "--draws=" => ""))
        else
            push!(labels, arg)
        end
    end

    all_samples = vcat(SAMPLES, SHOCK_SEGMENTS)
    samples = isempty(labels) ? all_samples : [s for s in all_samples if s.label in labels]
    if isempty(samples)
        println("No matching sample labels. Available labels:")
        for s in all_samples
            println("  ", s.label)
        end
        return
    end

    df_raw = CSV.read(joinpath(DERIVED_DIR, "hfi_core_data.csv"), DataFrame)
    df_raw.date = Date.(df_raw.date)

    println("Data: ", joinpath(DERIVED_DIR, "hfi_core_data.csv"))
    println("Samples: ", join([s.label for s in samples], ", "))
    println("Posterior draws per actual run: ", n_draws)

    for s in samples
        run_for_sample(df_raw, s; n_draws=n_draws)
    end
end

main()
