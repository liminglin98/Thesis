##
# =============================================================================
# RR SHOCKS DIAGNOSTICS
# =============================================================================
#
# Usage:
#   julia src/julia/RRShocks_diagnostics.jl
#
# Optional sample labels:
#   julia src/julia/RRShocks_diagnostics.jl 2025 2022 seg_2020_2025
#
# Optional FX variants:
#   julia src/julia/RRShocks_diagnostics.jl --fx=current 2025
#   julia src/julia/RRShocks_diagnostics.jl --fx=inverted 2025
#   julia src/julia/RRShocks_diagnostics.jl --fx=both 2025
#
# This script mirrors the shock-estimation part of RRShocks_monthly.jl, but it
# does not save files or run posterior IRF simulations. It prints diagnostics for
# the policy-rule residuals, the BVAR estimation sample, and the IV first stage.
# =============================================================================

using CSV, DataFrames
using Dates
using GLM, StatsModels
using LinearAlgebra
using Plots
using Printf
using Random
using ShiftedArrays
using Statistics

include(joinpath(@__DIR__, "common.jl"))

const RR_VAR_SYMS = [:realgdp_monthly_yoy, :cpi, :FR007, :neer_yoy, :IP_yoy]
const RR_VAR_LABELS = ["Real GDP Growth", "CPI", "FR007", "NEER YoY", "IP YoY"]
const RR_POLICY_COLS = [:FR007, :FR007_l1, :cpi_gap_l1, :gap_pos_l1, :gap_neg_l1, :fx_gap_l1]
const RR_DIFF_POLICY_COLS = [:dFR007, :cpi_gap_l1, :gap_pos_l1, :gap_neg_l1, :fx_gap_l1]

rr_policy_formula() = @formula(FR007 ~ FR007_l1 + cpi_gap_l1 + gap_pos_l1 + gap_neg_l1 + fx_gap_l1)
rr_diff_policy_formula() = @formula(dFR007 ~ cpi_gap_l1 + gap_pos_l1 + gap_neg_l1 + fx_gap_l1)

function finite_values(v)
    out = Float64[]
    for x in skipmissing(v)
        xf = Float64(x)
        isfinite(xf) && push!(out, xf)
    end
    return out
end

function fmt_float(x; digits::Int=4)
    if x === nothing || ismissing(x) || !isfinite(Float64(x))
        return "NA"
    end
    return @sprintf("%.*f", digits, Float64(x))
end

function print_vector_summary(name::AbstractString, v)
    vals = finite_values(v)
    println("\n-- $name --")
    println("  nonmissing finite: ", length(vals))
    if isempty(vals)
        return
    end
    println("  mean/sd/min/max: ",
        fmt_float(mean(vals)), " / ",
        fmt_float(std(vals)), " / ",
        fmt_float(minimum(vals)), " / ",
        fmt_float(maximum(vals)))
    println("  q05/q25/q50/q75/q95: ",
        fmt_float(quantile(vals, 0.05)), " / ",
        fmt_float(quantile(vals, 0.25)), " / ",
        fmt_float(quantile(vals, 0.50)), " / ",
        fmt_float(quantile(vals, 0.75)), " / ",
        fmt_float(quantile(vals, 0.95)))
    println("  near-zero abs < 1e-8: ", count(abs.(vals) .< 1e-8))
end

function print_float_quantiles(name::AbstractString, vals::Vector{Float64})
    vals = filter(isfinite, vals)
    println("\n-- $name --")
    println("  n: ", length(vals))
    if isempty(vals)
        return
    end
    println("  mean/sd/min/max: ",
        fmt_float(mean(vals), digits=6), " / ",
        fmt_float(std(vals), digits=6), " / ",
        fmt_float(minimum(vals), digits=6), " / ",
        fmt_float(maximum(vals), digits=6))
    println("  q50/q90/q95/q99: ",
        fmt_float(quantile(vals, 0.50), digits=6), " / ",
        fmt_float(quantile(vals, 0.90), digits=6), " / ",
        fmt_float(quantile(vals, 0.95), digits=6), " / ",
        fmt_float(quantile(vals, 0.99), digits=6))
end

function print_missingness(df::DataFrame, cols::Vector{Symbol})
    println("\nMissingness")
    for c in cols
        nmiss = count(ismissing, df[!, c])
        println(@sprintf("  %-22s %4d / %4d", String(c), nmiss, nrow(df)))
    end
end

function print_top_abs_shocks(df::DataFrame; nshow::Int=12)
    tmp = dropmissing(df[:, [:date, :policy_residual, :FR007, :pred_FR007]])
    if nrow(tmp) == 0
        println("\nLargest abs policy residuals: none")
        return
    end
    tmp.abs_resid = abs.(tmp.policy_residual)
    sort!(tmp, :abs_resid, rev=true)

    println("\nLargest abs policy residuals")
    println(@sprintf("  %-12s %12s %12s %12s", "date", "residual", "FR007", "pred"))
    for r in eachrow(first(tmp, min(nshow, nrow(tmp))))
        println(@sprintf("  %-12s %+12.5f %12.5f %12.5f",
            string(r.date), r.policy_residual, r.FR007, r.pred_FR007))
    end
end

function print_policy_coefficients(model)
    names = coefnames(model)
    beta = coef(model)
    se = stderror(model)
    tvals = coeftable(model).cols[3]
    pvals = coeftable(model).cols[4]

    labels = Dict(
        "(Intercept)" => "Intercept",
        "FR007_l1" => "Interest-rate smoothing: FR007 lag",
        "dFR007" => "Change in FR007",
        "cpi_gap_l1" => "Inflation gap lag",
        "gap_pos_l1" => "Output gap lag, target met",
        "gap_neg_l1" => "Output gap lag, target missed",
        "fx_gap_l1" => "FX pressure gap lag",
    )

    println(@sprintf("  %-38s %12s %12s %10s %10s", "Term", "coef", "std.err", "t", "p"))
    for i in eachindex(names)
        label = get(labels, names[i], names[i])
        println(@sprintf("  %-38s %+12.6f %12.6f %10.3f %10.4f",
            label, beta[i], se[i], tvals[i], pvals[i]))
    end
end

function print_policy_contributions(df::DataFrame, model, complete)
    model_df = df[complete, :]
    names = coefnames(model)
    beta = coef(model)

    term_specs = [
        ("Interest-rate smoothing", "FR007_l1", :FR007_l1),
        ("Inflation gap", "cpi_gap_l1", :cpi_gap_l1),
        ("Output gap, target met", "gap_pos_l1", :gap_pos_l1),
        ("Output gap, target missed", "gap_neg_l1", :gap_neg_l1),
        ("FX pressure gap", "fx_gap_l1", :fx_gap_l1),
    ]

    println("\nPolicy-rule contribution sizes")
    println(@sprintf("  %-28s %12s %12s %12s %12s", "Part", "coef", "x sd", "contrib sd", "mean contrib"))
    for (label, name, col) in term_specs
        idx = findfirst(==(name), names)
        isnothing(idx) && continue
        x = Vector{Float64}(model_df[:, col])
        contrib = beta[idx] .* x
        println(@sprintf("  %-28s %+12.6f %12.6f %12.6f %+12.6f",
            label, beta[idx], std(x), std(contrib), mean(contrib)))
    end
end

function first_stage_with_intercept(z::Vector{Float64}, u_policy::Vector{Float64})
    X = hcat(ones(length(z)), z)
    beta = X \ u_policy
    u_hat = X * beta
    resid = u_policy - u_hat
    sse = sum(resid .^ 2)
    sst = sum((u_policy .- mean(u_policy)) .^ 2)
    r2 = 1.0 - sse / sst
    sigma2 = sse / (length(z) - size(X, 2))
    vcov = sigma2 .* inv(X' * X)
    se_z = sqrt(vcov[2, 2])
    t_z = beta[2] / se_z
    f_z = t_z^2
    return (alpha=beta[1], gamma=beta[2], r2=r2, f=f_z, t=t_z, se=se_z)
end

function no_intercept_first_stage(z::Vector{Float64}, u_policy::Vector{Float64})
    gamma = dot(z, u_policy) / dot(z, z)
    u_hat = z .* gamma
    sse = sum((u_policy .- u_hat) .^ 2)
    sst_centered = sum((u_policy .- mean(u_policy)) .^ 2)
    r2_centered = 1.0 - sse / sst_centered
    f_centered = r2_centered / (1.0 - r2_centered) * (length(z) - 2)
    r2_uncentered = sum(u_hat .^ 2) / sum(u_policy .^ 2)
    f_uncentered = r2_uncentered / (1.0 - r2_uncentered) * (length(z) - 1)
    return (gamma=gamma, r2_centered=r2_centered, f_centered=f_centered,
            r2_uncentered=r2_uncentered, f_uncentered=f_uncentered)
end

function companion_max_root(B::Matrix{Float64}, n::Int, p::Int)
    A = get_lag_matrices(B, n, p)
    comp = zeros(n * p, n * p)
    comp[1:n, :] = hcat(A...)
    if p > 1
        comp[n+1:end, 1:n*(p-1)] = Matrix(I, n * (p - 1), n * (p - 1))
    end
    return maximum(abs.(eigvals(comp)))
end

function posterior_irf_draws(B_post::Matrix{Float64}, V_post::Vector{Matrix{Float64}},
                             Y_dep::Matrix{Float64}, X_var::Matrix{Float64}, z::Vector{Float64},
                             policy_col::Int, gdp_col::Int, n::Int, p::Int, H::Int;
                             n_draws::Int=5000, stability_filter::Bool=false,
                             sample_mask=nothing, impact_scale::Float64=1.0)
    k = size(X_var, 2)
    irf_draws = zeros(n_draws, H + 1, n)
    roots = Float64[]
    n_valid = 0
    n_unstable = 0
    n_iv_fail = 0
    n_huge_irf = 0

    Random.seed!(42)
    for _ in 1:n_draws
        B_draw = zeros(k, n)
        for j in 1:n
            C_cov = cholesky(Symmetric(V_post[j] + 1e-12 * I(k)); check=false)
            if !issuccess(C_cov)
                E = eigen(Symmetric(V_post[j]))
                sqrtV = E.vectors * Diagonal(sqrt.(max.(E.values, 1e-12)))
                B_draw[:, j] = B_post[:, j] + sqrtV * randn(k)
            else
                B_draw[:, j] = B_post[:, j] + C_cov.L * randn(k)
            end
        end

        root_draw = companion_max_root(B_draw, n, p)
        push!(roots, root_draw)
        if stability_filter && root_draw >= 1.0
            n_unstable += 1
            continue
        end

        U_draw = Y_dep - X_var * B_draw
        U_iv = sample_mask === nothing ? U_draw : U_draw[sample_mask, :]
        z_iv = sample_mask === nothing ? z : z[sample_mask]
        b_draw, _, _ = iv_identify(U_iv, z_iv, policy_col, gdp_col)
        if isnothing(b_draw)
            n_iv_fail += 1
            continue
        end
        b_draw .*= impact_scale

        A_draw = get_lag_matrices(B_draw, n, p)
        C_draw = compute_ma(A_draw, n, p, H)
        irf_draw = compute_irfs(C_draw, b_draw, H)

        if any(abs.(irf_draw) .> 500)
            n_huge_irf += 1
            continue
        end

        n_valid += 1
        irf_draws[n_valid, :, :] = irf_draw
    end

    return (
        irf_draws=irf_draws[1:max(n_valid, 1), :, :],
        roots=roots,
        n_valid=n_valid,
        n_unstable=n_unstable,
        n_iv_fail=n_iv_fail,
        n_huge_irf=n_huge_irf,
        n_draws=n_draws,
    )
end

function irf_bands(irf_draws::Array{Float64, 3}, H::Int, n::Int)
    median = zeros(H + 1, n)
    lo68 = zeros(H + 1, n)
    hi68 = zeros(H + 1, n)
    lo90 = zeros(H + 1, n)
    hi90 = zeros(H + 1, n)

    for h in 0:H, j in 1:n
        draws = irf_draws[:, h+1, j]
        median[h+1, j] = quantile(draws, 0.50)
        lo68[h+1, j] = quantile(draws, 0.16)
        hi68[h+1, j] = quantile(draws, 0.84)
        lo90[h+1, j] = quantile(draws, 0.05)
        hi90[h+1, j] = quantile(draws, 0.95)
    end

    return (median=median, lo68=lo68, hi68=hi68, lo90=lo90, hi90=hi90)
end

function save_irf_plot(irf_point::Matrix{Float64}, draws_result, H::Int,
                       sample_label::String, fx_variant::Symbol, suffix::String,
                       title_suffix::String)
    n = size(irf_point, 2)
    bands = irf_bands(draws_result.irf_draws, H, n)

    p_plots = []
    for j in 1:n
        plt = plot(0:H, bands.median[:, j],
            title=RR_VAR_LABELS[j], label="Median",
            xlabel="Months", ylabel="Response",
            legend=:best, linewidth=2.5, color=:darkblue)
        plot!(plt, 0:H, bands.lo90[:, j],
            fillrange=bands.hi90[:, j], fillalpha=0.18,
            fillcolor=:lightblue, linealpha=0, label="90% CS")
        plot!(plt, 0:H, bands.lo68[:, j],
            fillrange=bands.hi68[:, j], fillalpha=0.35,
            fillcolor=:steelblue, linealpha=0, label="68% CS")
        plot!(plt, 0:H, irf_point[:, j],
            label="Point", color=:black, linestyle=:dash, linewidth=1.5)
        hline!([0], color=:gray, linestyle=:dash, alpha=0.5, label="")
        push!(p_plots, plt)
    end

    n_cols = 3
    n_rows = ceil(Int, n / n_cols)
    fig = plot(p_plots...,
        layout=(n_rows, n_cols),
        size=(380 * n_cols, 300 * n_rows),
        plot_title="RR diagnostics IRFs: $(title_suffix) ($(sample_label), fx=$(fx_variant))")

    out_dir = diagnostics_dir(sample_label)
    out_path = joinpath(out_dir, "rr_irf_$(suffix)_fx_$(fx_variant).png")
    savefig(fig, out_path)
    println("  Saved IRF plot: ", out_path)
    return out_path
end

function run_irf_block(B_post::Matrix{Float64}, V_post::Vector{Matrix{Float64}},
                       Y_dep::Matrix{Float64}, X_var::Matrix{Float64}, z::Vector{Float64},
                       policy_col::Int, gdp_col::Int, n::Int, p::Int, H_irf::Int,
                       sample_label::String, fx_variant::Symbol, suffix::String,
                       title_suffix::String; stability_filter::Bool=true,
                       sample_mask=nothing, impact_scale::Float64=1.0)
    U_post = Y_dep - X_var * B_post
    U_iv = sample_mask === nothing ? U_post : U_post[sample_mask, :]
    z_iv = sample_mask === nothing ? z : z[sample_mask]

    if length(z_iv) <= 2
        println("  Not enough sign-selected observations; skipping $(title_suffix) IRF plot.")
        return nothing
    end

    b_point, F_stat, gamma_hat = iv_identify(U_iv, z_iv, policy_col, gdp_col)

    println("  IV gamma: ", fmt_float(gamma_hat, digits=6),
        ", F=", fmt_float(F_stat, digits=3),
        ", corr(z, u_policy)=", fmt_float(cor(z_iv, U_iv[:, policy_col]), digits=6),
        ", n=", length(z_iv))

    if isnothing(b_point)
        println("  IV identification failed; skipping $(title_suffix) IRF plot.")
        return nothing
    end
    b_point .*= impact_scale

    A_post = get_lag_matrices(B_post, n, p)
    C_post = compute_ma(A_post, n, p, H_irf)
    irf_point = compute_irfs(C_post, b_point, H_irf)

    draws = posterior_irf_draws(B_post, V_post, Y_dep, X_var, z,
        policy_col, gdp_col, n, p, H_irf; n_draws=5000,
        stability_filter=stability_filter, sample_mask=sample_mask,
        impact_scale=impact_scale)

    println("    valid draws: ", draws.n_valid, " / ", draws.n_draws)
    if stability_filter
        println("    unstable skipped: ", draws.n_unstable,
            ", huge-IRF skipped: ", draws.n_huge_irf,
            ", IV failures: ", draws.n_iv_fail)
    else
        println("    huge-IRF skipped: ", draws.n_huge_irf,
            ", IV failures: ", draws.n_iv_fail)
    end

    save_irf_plot(irf_point, draws, H_irf, sample_label, fx_variant, suffix, title_suffix)
    return (irf_point=irf_point, draws=draws, F_stat=F_stat, gamma_hat=gamma_hat)
end

function add_fx_gap!(df_raw::DataFrame, fx_variant::Symbol)
    df_raw.after2006 = coalesce.(df_raw.date .>= Date(2006, 1, 1), false)

    if fx_variant == :current
        df_raw.dCNYUSDCPR = df_raw.CNYUSDCPR .- lag(df_raw.CNYUSDCPR, 1)
        df_raw.fx_gap = df_raw.dCNYUSDCPR .* df_raw.after2006 .* (df_raw.CNYUSDSpot .- df_raw.CNYUSDCPR)
    elseif fx_variant == :inverted
        df_raw.CNYUSDSpot_inv = 1.0 ./ df_raw.CNYUSDSpot
        df_raw.CNYUSDCPR_inv = 1.0 ./ df_raw.CNYUSDCPR
        df_raw.dCNYUSDCPR_inv = df_raw.CNYUSDCPR_inv .- lag(df_raw.CNYUSDCPR_inv, 1)
        df_raw.fx_gap = df_raw.dCNYUSDCPR_inv .* df_raw.after2006 .* (df_raw.CNYUSDSpot_inv .- df_raw.CNYUSDCPR_inv)
    else
        error("Unknown fx_variant=$(fx_variant). Use :current or :inverted.")
    end

    df_raw.fx_gap_l1 = lag(df_raw.fx_gap, 1)
    return df_raw
end

function prepare_rr_data(fx_variant::Symbol)
    df_raw = CSV.read(joinpath(DERIVED_DIR, "romer_china_data.csv"), DataFrame)
    df_raw.date = Date.(df_raw.date)

    df_raw.gdp_gap = df_raw.realgdp_monthly_yoy - df_raw.target_gdp
    df_raw.cpi_gap = df_raw.cpi - df_raw.target_cpi

    df_raw.not_meet_target = df_raw.gdp_gap .< 1
    df_raw.gap_pos = df_raw.gdp_gap .* (1 .- df_raw.not_meet_target)
    df_raw.gap_neg = df_raw.gdp_gap .* df_raw.not_meet_target

    add_fx_gap!(df_raw, fx_variant)

    df_raw.FR007_l1   = lag(df_raw.FR007, 1)
    df_raw.dFR007     = df_raw.FR007 .- lag(df_raw.FR007, 1)
    df_raw.cpi_gap_l1 = lag(df_raw.cpi_gap, 1)
    df_raw.gap_pos_l1 = lag(df_raw.gap_pos, 1)
    df_raw.gap_neg_l1 = lag(df_raw.gap_neg, 1)

    return df_raw
end

function run_rr_diagnostics_for_sample(df_raw::DataFrame, s; p::Int=6, fx_variant::Symbol=:current)
    println("\n", "="^78)
    println("RR shock diagnostics: $(s.start_date) to $(s.end_date) [$(s.label)]")
    println("FX variant: ", fx_variant == :current ? "current USD/CNY term" : "Option B inverted CNY/USD-style term")
    println("="^78)

    df = filter(r -> !ismissing(r.date) && r.date >= s.start_date && r.date <= s.end_date, df_raw)
    println("Raw sample rows: ", nrow(df))
    if nrow(df) == 0
        println("No rows in sample.")
        return
    end
    println("Date range in data: ", minimum(df.date), " to ", maximum(df.date))

    print_missingness(df, unique(vcat([:date], RR_POLICY_COLS, RR_VAR_SYMS,
        RR_DIFF_POLICY_COLS, [:target_gdp, :target_cpi, :gdp_gap, :cpi_gap, :fx_gap])))

    complete = completecases(df[:, RR_POLICY_COLS])
    println("\nPolicy-rule complete rows: ", count(complete), " / ", nrow(df))
    if count(complete) <= length(RR_POLICY_COLS)
        println("Not enough complete rows for policy-rule regression.")
        return
    end

    model = lm(rr_policy_formula(), df[complete, :])
    println("\nPolicy-rule regression")
    print_policy_coefficients(model)
    println("  r2 / adjr2: ", fmt_float(r2(model)), " / ", fmt_float(adjr2(model)))
    println("  sigma: ", fmt_float(dof_residual(model) > 0 ? sqrt(deviance(model) / dof_residual(model)) : NaN))
    print_policy_contributions(df, model, complete)

    df.pred_FR007 = Vector{Union{Missing, Float64}}(missing, nrow(df))
    df.policy_residual = Vector{Union{Missing, Float64}}(missing, nrow(df))
    df.pred_FR007[complete] = predict(model)
    df.policy_residual[complete] = GLM.residuals(model)

    print_vector_summary("Policy residual", df.policy_residual)
    print_top_abs_shocks(df)

    diff_complete = completecases(df[:, RR_DIFF_POLICY_COLS])
    println("\nAlternative policy-rule complete rows (dFR007 lhs, no FR007 lag rhs): ",
        count(diff_complete), " / ", nrow(df))
    if count(diff_complete) > length(RR_DIFF_POLICY_COLS)
        diff_model = lm(rr_diff_policy_formula(), df[diff_complete, :])
        println("\nAlternative policy-rule regression: dFR007 ~ gaps + FX, no lagged FR007")
        print_policy_coefficients(diff_model)
        println("  r2 / adjr2: ", fmt_float(r2(diff_model)), " / ", fmt_float(adjr2(diff_model)))
        println("  sigma: ", fmt_float(dof_residual(diff_model) > 0 ? sqrt(deviance(diff_model) / dof_residual(diff_model)) : NaN))

        df.pred_dFR007 = Vector{Union{Missing, Float64}}(missing, nrow(df))
        df.policy_residual_dfr007_nolag = Vector{Union{Missing, Float64}}(missing, nrow(df))
        df.pred_dFR007[diff_complete] = predict(diff_model)
        df.policy_residual_dfr007_nolag[diff_complete] = GLM.residuals(diff_model)
        print_vector_summary("Alternative residual: dFR007 lhs, no FR007 lag rhs",
            df.policy_residual_dfr007_nolag)
    else
        println("Not enough complete rows for alternative dFR007 no-lag policy-rule regression.")
        df.policy_residual_dfr007_nolag = Vector{Union{Missing, Float64}}(missing, nrow(df))
    end

    all_syms = vcat(RR_VAR_SYMS, [:policy_residual, :policy_residual_dfr007_nolag])
    df_bvar = dropmissing(df, all_syms)
    println("\nBVAR/IV alignment")
    println("  complete rows after VAR vars + residual: ", nrow(df_bvar), " / ", nrow(df))
    if nrow(df_bvar) <= p + 5
        println("Not enough rows after BVAR dropmissing for p=$p.")
        return
    end
    println("  date range after dropmissing: ", first(df_bvar.date), " to ", last(df_bvar.date))

    n = length(RR_VAR_SYMS)
    Y_full = Matrix{Float64}(df_bvar[:, RR_VAR_SYMS])
    z_full = Vector{Float64}(df_bvar[:, :policy_residual])
    z_alt_full = Vector{Float64}(df_bvar[:, :policy_residual_dfr007_nolag])
    T_raw = size(Y_full, 1)
    Teff = T_raw - p
    Y_dep = Y_full[p+1:end, :]
    X_var = ones(Teff, 1 + n * p)
    for L in 1:p
        X_var[:, 1 + (L-1)*n+1 : 1 + L*n] = Y_full[p+1-L:end-L, :]
    end
    z = z_full[p+1:end]
    z_alt = z_alt_full[p+1:end]
    dates_est = df_bvar.date[p+1:end]

    println("  VAR lag p: ", p)
    println("  effective IV sample: ", Teff, " rows, ", first(dates_est), " to ", last(dates_est))
    print_vector_summary("Aligned instrument z = policy residual", z)
    print_vector_summary("Aligned alternative z = dFR007 residual, no FR007 lag", z_alt)

    B_ols = X_var \ Y_dep
    U_ols = Y_dep - X_var * B_ols
    sigma2_ols = vec(var(U_ols, dims=1))
    println("\nVAR OLS residual sd by equation")
    for (j, sym) in enumerate(RR_VAR_SYMS)
        println(@sprintf("  %-22s %10.5f", String(sym), sqrt(sigma2_ols[j])))
    end

    λ = 0.2
    d = 1.0
    k = size(X_var, 2)
    B0 = zeros(k, n)
    for j in 1:n
        B0[1 + j, j] = 1.0
    end

    Omega_diag = zeros(k, n)
    for j in 1:n
        Omega_diag[1, j] = 100.0 * sigma2_ols[j]
        for L in 1:p, i in 1:n
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
    policy_col = findfirst(==(:FR007), RR_VAR_SYMS)
    gdp_col = findfirst(==(:realgdp_monthly_yoy), RR_VAR_SYMS)
    u_policy = U_post[:, policy_col]

    println("\nPosterior-mean VAR diagnostics")
    root_post = companion_max_root(B_post, n, p)
    println("  max companion root: ", fmt_float(root_post, digits=6),
        root_post >= 1.0 ? "  UNSTABLE" : root_post >= 0.99 ? "  near unit root" : "")
    println("  corr(z, u_policy): ", fmt_float(cor(z, u_policy), digits=6))
    println("  cov(z, u_policy):  ", fmt_float(cov(z, u_policy), digits=6))
    println("  dot(z,z):          ", fmt_float(dot(z, z), digits=6))

    fs_int = first_stage_with_intercept(z, u_policy)
    fs_no = no_intercept_first_stage(z, u_policy)
    b_point, F_common, gamma_common = iv_identify(U_post, z, policy_col, gdp_col)

    println("\nFirst-stage comparison")
    println("  With intercept:    alpha=", fmt_float(fs_int.alpha, digits=6),
        " gamma=", fmt_float(fs_int.gamma, digits=6),
        " se=", fmt_float(fs_int.se, digits=6),
        " t=", fmt_float(fs_int.t, digits=3),
        " F=", fmt_float(fs_int.f, digits=3),
        " R2=", fmt_float(fs_int.r2, digits=6))
    println("  No intercept:      gamma=", fmt_float(fs_no.gamma, digits=6),
        " F centered=", fmt_float(fs_no.f_centered, digits=3),
        " R2 centered=", fmt_float(fs_no.r2_centered, digits=6),
        " F uncentered=", fmt_float(fs_no.f_uncentered, digits=3),
        " R2 uncentered=", fmt_float(fs_no.r2_uncentered, digits=6))
    println("  common.jl output:  gamma=", fmt_float(gamma_common, digits=6),
        " F=", fmt_float(F_common, digits=3))

    if isnothing(b_point)
        println("\nIV impact vector: identification failed")
        println("Skipping IRF plot generation.")
        return
    else
        println("\nIV impact vector, normalized to +1 FR007")
        for (j, sym) in enumerate(RR_VAR_SYMS)
            println(@sprintf("  %-22s %+12.6f", String(sym), b_point[j]))
        end
    end

    H_irf = 24
    A_post = get_lag_matrices(B_post, n, p)
    C_post = compute_ma(A_post, n, p, H_irf)
    irf_point = compute_irfs(C_post, b_point, H_irf)

    println("\nGenerating IRF diagnostic plots")
    println("  Existing draw logic: no companion-root filter, abs(IRF)>500 safeguard retained")
    existing_irfs = posterior_irf_draws(B_post, V_post, Y_dep, X_var, z,
        policy_col, gdp_col, n, p, H_irf; n_draws=5000, stability_filter=false)
    println("    valid draws: ", existing_irfs.n_valid, " / ", existing_irfs.n_draws)
    println("    huge-IRF skipped: ", existing_irfs.n_huge_irf, ", IV failures: ", existing_irfs.n_iv_fail)
    save_irf_plot(irf_point, existing_irfs, H_irf, s.label, fx_variant,
        "existing_no_stability_filter", "existing draw logic")

    println("  Stability-filtered draw logic: discard max companion root >= 1.0 before IRFs")
    filtered_irfs = posterior_irf_draws(B_post, V_post, Y_dep, X_var, z,
        policy_col, gdp_col, n, p, H_irf; n_draws=5000, stability_filter=true)
    println("    valid draws: ", filtered_irfs.n_valid, " / ", filtered_irfs.n_draws)
    println("    unstable skipped: ", filtered_irfs.n_unstable,
        ", huge-IRF skipped: ", filtered_irfs.n_huge_irf,
        ", IV failures: ", filtered_irfs.n_iv_fail)
    save_irf_plot(irf_point, filtered_irfs, H_irf, s.label, fx_variant,
        "stability_filtered", "stability filtered")

    println("  Alternative dFR007 no-lag shock: stability-filtered final IRFs")
    run_irf_block(B_post, V_post, Y_dep, X_var, z_alt,
        policy_col, gdp_col, n, p, H_irf, s.label, fx_variant,
        "dfr007_nolag_stability_filtered", "dFR007 lhs, no FR007 lag rhs";
        stability_filter=true)

    println("  Alternative dFR007 no-lag shock: contractionary sign-selected IRFs (+1 pp FR007)")
    pos_mask = z_alt .> 0.0
    run_irf_block(B_post, V_post, Y_dep, X_var, z_alt,
        policy_col, gdp_col, n, p, H_irf, s.label, fx_variant,
        "dfr007_nolag_contractionary_pos_stability_filtered",
        "dFR007 no-lag, contractionary shocks (+1 pp FR007)";
        stability_filter=true, sample_mask=pos_mask, impact_scale=1.0)

    println("  Alternative dFR007 no-lag shock: expansionary sign-selected IRFs (-1 pp FR007)")
    neg_mask = z_alt .< 0.0
    run_irf_block(B_post, V_post, Y_dep, X_var, z_alt,
        policy_col, gdp_col, n, p, H_irf, s.label, fx_variant,
        "dfr007_nolag_expansionary_neg_stability_filtered",
        "dFR007 no-lag, expansionary shocks (-1 pp FR007)";
        stability_filter=true, sample_mask=neg_mask, impact_scale=-1.0)

    println("\nPosterior draw stability check")
    n_check_draws = 1000
    H_check = 24
    roots = Float64[]
    max_abs_irfs = Float64[]
    n_unstable_999 = 0
    n_unstable_1000 = 0
    n_iv_fail = 0
    n_huge_irf = 0

    Random.seed!(42)
    for _ in 1:n_check_draws
        B_draw = zeros(k, n)
        for j in 1:n
            C_cov = cholesky(Symmetric(V_post[j] + 1e-12 * I(k)); check=false)
            if !issuccess(C_cov)
                E = eigen(Symmetric(V_post[j]))
                sqrtV = E.vectors * Diagonal(sqrt.(max.(E.values, 1e-12)))
                B_draw[:, j] = B_post[:, j] + sqrtV * randn(k)
            else
                B_draw[:, j] = B_post[:, j] + C_cov.L * randn(k)
            end
        end

        root_draw = companion_max_root(B_draw, n, p)
        push!(roots, root_draw)
        root_draw >= 0.999 && (n_unstable_999 += 1)
        root_draw >= 1.0 && (n_unstable_1000 += 1)
        root_draw >= 0.999 && continue

        U_draw = Y_dep - X_var * B_draw
        b_draw, _, _ = iv_identify(U_draw, z, policy_col, gdp_col)
        if isnothing(b_draw)
            n_iv_fail += 1
            continue
        end

        A_draw = get_lag_matrices(B_draw, n, p)
        C_draw = compute_ma(A_draw, n, p, H_check)
        irf_draw = compute_irfs(C_draw, b_draw, H_check)
        max_abs_irf = maximum(abs.(irf_draw))
        push!(max_abs_irfs, max_abs_irf)
        max_abs_irf > 500 && (n_huge_irf += 1)
    end

    print_float_quantiles("Max companion roots across posterior draws", roots)
    print_float_quantiles("Max abs IRF among stable identified draws", max_abs_irfs)
    println("  draws checked:             ", n_check_draws)
    println("  roots >= 0.999 skipped:    ", n_unstable_999)
    println("  roots >= 1.000 explosive:  ", n_unstable_1000)
    println("  IV failures after stable:  ", n_iv_fail)
    println("  max abs IRF > 500:         ", n_huge_irf)

    println("\nMonthly residual by year")
    tmp = dropmissing(df[:, [:date, :policy_residual]])
    if nrow(tmp) > 0
        tmp.year = year.(tmp.date)
        gd = combine(groupby(tmp, :year),
            :policy_residual => length => :n,
            :policy_residual => mean => :mean,
            :policy_residual => std => :sd,
            :policy_residual => (x -> maximum(abs.(x))) => :max_abs)
        for r in eachrow(gd)
            println(@sprintf("  %4d  n=%3d  mean=%+9.5f  sd=%9.5f  maxabs=%9.5f",
                r.year, r.n, r.mean, coalesce(r.sd, NaN), r.max_abs))
        end
    end
end

function main()
    all_samples = vcat(SAMPLES, SHOCK_SEGMENTS)

    fx_arg = "current"
    labels = String[]
    for arg in ARGS
        if startswith(arg, "--fx=")
            fx_arg = replace(arg, "--fx=" => "")
        else
            push!(labels, arg)
        end
    end

    fx_variants = if fx_arg == "current"
        [:current]
    elseif fx_arg == "inverted"
        [:inverted]
    elseif fx_arg == "both"
        [:current, :inverted]
    else
        error("Unknown --fx=$(fx_arg). Use current, inverted, or both.")
    end

    samples = isempty(labels) ? all_samples : [s for s in all_samples if s.label in labels]
    if isempty(samples)
        println("No matching sample labels. Available labels:")
        for s in all_samples
            println("  ", s.label)
        end
        return
    end

    println("Data: ", joinpath(DERIVED_DIR, "romer_china_data.csv"))
    println("Samples: ", join([s.label for s in samples], ", "))
    println("FX variants: ", join(string.(fx_variants), ", "))

    for fx_variant in fx_variants
        df_raw = prepare_rr_data(fx_variant)
        println("\nLoaded rows: ", nrow(df_raw), ", dates: ", minimum(skipmissing(df_raw.date)), " to ", maximum(skipmissing(df_raw.date)))
        for s in samples
            run_rr_diagnostics_for_sample(df_raw, s; fx_variant=fx_variant)
        end
    end
end

main()
