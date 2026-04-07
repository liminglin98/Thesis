##
using CSV, DataFrames, Statistics, GLM, StatsModels
using Plots, Dates, ShiftedArrays
using Printf, Random, Distributions, LinearAlgebra
using Serialization
##
include(joinpath(@__DIR__, "common.jl"))
##
# Load data once
df_raw = CSV.read(joinpath(DERIVED_DIR, "hfi_core_data.csv"), DataFrame)
df_raw.date = Date.(df_raw.date)

var_syms   = [:realgdp_monthly_yoy, :cpi, :FR007, :neer_yoy, :IP_yoy]
var_labels = ["Real GDP Growth", "CPI", "FR007", "Real Effective Exchange Rate", "Industrial Value Added"]

n          = length(var_syms)
p          = 6    # lags
H          = 24   # IRF horizon

gdp_col    = findfirst(==(:realgdp_monthly_yoy), var_syms)
policy_col = findfirst(==(:FR007), var_syms)

##
# ============================================================
# Main estimation function
# ============================================================

function run_hfi_bvar(df::DataFrame, instrument_sym::Symbol;
                      λ=0.2, d=1.0, n_draws=5000, label="")

    all_syms = vcat(var_syms, [instrument_sym])
    df_bvar  = dropmissing(df, all_syms)
    dates_bvar = df_bvar.date

    Y_full = Matrix{Float64}(df_bvar[:, var_syms])
    z_full = [ismissing(x) || isnan(x) ? 0.0 : Float64(x) for x in df_bvar[:, instrument_sym]]
    T_raw  = size(Y_full, 1)

    Teff  = T_raw - p
    Y_dep = Y_full[p+1:end, :]
    X_var = ones(Teff, 1 + n * p)
    for L in 1:p
        X_var[:, 1 + (L-1)*n+1 : 1 + L*n] = Y_full[p+1-L:end-L, :]
    end
    z         = z_full[p+1:end]
    dates_est = dates_bvar[p+1:end]
    k         = size(X_var, 2)

    println("\n" * "="^60)
    println("HFI BVAR+IV-SVAR  —  instrument: $(instrument_sym)  $(label)")
    println("Sample: $(dates_est[1]) — $(dates_est[end])  |  T=$(Teff), n=$(n), p=$(p)")
    println("="^60)

    # Minnesota prior
    B_ols      = X_var \ Y_dep
    U_ols      = Y_dep - X_var * B_ols
    sigma2_ols = vec(var(U_ols, dims=1))

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

    # Posterior
    B_post = zeros(k, n)
    V_post = Vector{Matrix{Float64}}(undef, n)
    for j in 1:n
        Omega_j_inv  = Diagonal(1.0 ./ Omega_diag[:, j])
        V_j_post_inv = Omega_j_inv + X_var' * X_var
        V_j_post     = Symmetric(inv(V_j_post_inv))
        B_post[:, j] = V_j_post * (Omega_j_inv * B0[:, j] + X_var' * Y_dep[:, j])
        V_post[j]    = Matrix(V_j_post)
    end

    U_post = Y_dep - X_var * B_post

    # IV identification (point estimate)
    b_point, F_stat, _ = iv_identify(U_post, z, policy_col, gdp_col)
    isnothing(b_point) && error("IV identification failed on posterior mean")

    println(@sprintf("First-stage F-stat:  %.2f  %s", F_stat, F_stat < 10 ? "⚠ WEAK" : "✓ OK"))
    println(@sprintf("Corr(z, u_policy):   %.4f", cor(z, U_post[:, policy_col])))

    A_post = get_lag_matrices(B_post, n, p)
    C_post = compute_ma(A_post, n, p, H)
    irf    = compute_irfs(C_post, b_point, H)

    # Posterior draws
    Random.seed!(42)
    irf_draws = zeros(n_draws, H + 1, n)
    n_valid   = Ref(0)

    for _ in 1:n_draws
        B_draw = zeros(k, n)
        for j in 1:n
            C_cov = cholesky(Symmetric(V_post[j] + 1e-12 * I(k)); check=false)
            if !issuccess(C_cov)
                E     = eigen(Symmetric(V_post[j]))
                sqrtV = E.vectors * Diagonal(sqrt.(max.(E.values, 1e-12)))
                B_draw[:, j] = B_post[:, j] + sqrtV * randn(k)
            else
                B_draw[:, j] = B_post[:, j] + C_cov.L * randn(k)
            end
        end

        U_draw       = Y_dep - X_var * B_draw
        b_draw, _, _ = iv_identify(U_draw, z, policy_col, gdp_col)
        isnothing(b_draw) && continue

        A_draw   = get_lag_matrices(B_draw, n, p)
        C_draw   = compute_ma(A_draw, n, p, H)
        irf_draw = compute_irfs(C_draw, b_draw, H)
        any(abs.(irf_draw) .> 500) && continue

        n_valid[] += 1
        irf_draws[n_valid[], :, :] = irf_draw
    end

    valid_draws = n_valid[]
    irf_draws   = irf_draws[1:max(valid_draws, 1), :, :]
    println(@sprintf("Posterior draws: %d / %d valid", valid_draws, n_draws))

    # Credible sets
    irf_median = zeros(H + 1, n)
    irf_68_lo  = zeros(H + 1, n)
    irf_68_hi  = zeros(H + 1, n)
    irf_90_lo  = zeros(H + 1, n)
    irf_90_hi  = zeros(H + 1, n)

    if valid_draws > 1
        for h in 0:H, j in 1:n
            draws = irf_draws[:, h+1, j]
            irf_median[h+1, j] = quantile(draws, 0.50)
            irf_68_lo[h+1, j]  = quantile(draws, 0.16)
            irf_68_hi[h+1, j]  = quantile(draws, 0.84)
            irf_90_lo[h+1, j]  = quantile(draws, 0.05)
            irf_90_hi[h+1, j]  = quantile(draws, 0.95)
        end
    end

    # IRF table
    println("\n" * "="^60)
    println("IRFs — Contractionary MP Shock (+1 pp FR007)")
    println("Instrument: $(instrument_sym)")
    println("="^60)
    for j in 1:n
        println("\n--- $(var_labels[j]) ---")
        println(@sprintf("  %4s  %8s  %8s  [%8s, %8s]  [%8s, %8s]",
            "h", "Median", "Point", "68lo", "68hi", "90lo", "90hi"))
        for h in [0, 1, 3, 6, 9, 12, 18, 24]
            h > H && continue
            println(@sprintf("  %4d  %+8.4f  %+8.4f  [%+8.4f, %+8.4f]  [%+8.4f, %+8.4f]",
                h, irf_median[h+1,j], irf[h+1,j],
                irf_68_lo[h+1,j], irf_68_hi[h+1,j],
                irf_90_lo[h+1,j], irf_90_hi[h+1,j]))
        end
    end

    return (irf=irf, irf_median=irf_median,
            irf_68_lo=irf_68_lo, irf_68_hi=irf_68_hi,
            irf_90_lo=irf_90_lo, irf_90_hi=irf_90_hi,
            irf_draws=irf_draws,
            F_stat=F_stat, valid_draws=valid_draws)
end

##
# ============================================================
# Plot helper
# ============================================================

function plot_irfs(res, title_suffix, color)
    p_plots = []
    for j in 1:n
        plt = plot(0:H, res.irf_median[:, j],
            title=var_labels[j], label="Median",
            xlabel="Months", ylabel="Response",
            legend=:best, linewidth=2.5, color=color)
        plot!(plt, 0:H, res.irf_90_lo[:, j],
            fillrange=res.irf_90_hi[:, j],
            fillalpha=0.15, fillcolor=color, linealpha=0, label="90% CS")
        plot!(plt, 0:H, res.irf_68_lo[:, j],
            fillrange=res.irf_68_hi[:, j],
            fillalpha=0.35, fillcolor=color, linealpha=0, label="68% CS")
        hline!([0], color=:gray, linestyle=:dash, alpha=0.5, label="")
        push!(p_plots, plt)
    end
    n_cols = 3
    fig = plot(p_plots...,
        layout=(ceil(Int, n/n_cols), n_cols),
        size=(380*n_cols, 300*ceil(Int, n/n_cols)),
        plot_title="BVAR+IV-SVAR: +1 pp FR007 shock\n$(title_suffix)")
    display(fig)
    return fig
end

##
# ============================================================
# Loop over sample periods
# ============================================================

for s in SAMPLES

println("\n", "="^70)
println("  HFIShocks — Sample: $(s.start_date) to $(s.end_date)  [$(s.label)]")
println("="^70)

MAIN_DIR   = main_results_dir(s.label)
ROBUST_DIR = robustness_dir(s.label)
INTER_DIR  = intermediate_dir(s.label)

# Filter by date range
df = filter(r -> !ismissing(r.date) && r.date >= s.start_date && r.date <= s.end_date, df_raw)

# Run both instruments
res_policy        = run_hfi_bvar(df, :shock_policy;        label="(any announcement) [$(s.label)]")
res_policy_change = run_hfi_bvar(df, :shock_policy_change; label="(rate change only) [$(s.label)]")

# Plots
fig1 = plot_irfs(res_policy,        "HFI: any policy announcement ($(s.label))", :darkblue)
fig2 = plot_irfs(res_policy_change, "HFI: rate change only ($(s.label))",         :darkred)
savefig(fig1, joinpath(ROBUST_DIR, "irf_hfi_shock_policy.png"))
savefig(fig2, joinpath(MAIN_DIR, "irf_hfi_shock_policy_change.png"))

# Overlay comparison
p_compare = []
for j in 1:n
    plt = plot(0:H, res_policy.irf_median[:, j],
        label="Any announcement", color=:darkblue, linewidth=2)
    plot!(plt, 0:H, res_policy.irf_68_lo[:, j],
        fillrange=res_policy.irf_68_hi[:, j],
        fillalpha=0.2, fillcolor=:darkblue, linealpha=0, label="")
    plot!(plt, 0:H, res_policy_change.irf_median[:, j],
        label="Rate change only", color=:darkred, linewidth=2, linestyle=:dash)
    plot!(plt, 0:H, res_policy_change.irf_68_lo[:, j],
        fillrange=res_policy_change.irf_68_hi[:, j],
        fillalpha=0.2, fillcolor=:darkred, linealpha=0, label="")
    hline!([0], color=:gray, linestyle=:dot, alpha=0.5, label="")
    plot!(plt, title=var_labels[j], xlabel="Months", ylabel="Response", legend=:best)
    push!(p_compare, plt)
end

n_cols = 3
fig_compare = plot(p_compare...,
    layout=(ceil(Int, n/n_cols), n_cols),
    size=(380*n_cols, 300*ceil(Int, n/n_cols)),
    plot_title="HFI Shock Comparison ($(s.label))")
display(fig_compare)
savefig(fig_compare, joinpath(MAIN_DIR, "irf_hfi_comparison.png"))

# Diagnostics summary
println("\n" * "="^60)
println("Diagnostics Summary [$(s.label)]")
println("="^60)
println(@sprintf("%-30s  %10s  %10s", "Metric", "Any ann.", "Rate chg"))
println(@sprintf("%-30s  %10.2f  %10.2f", "First-stage F-stat",
    res_policy.F_stat, res_policy_change.F_stat))
println(@sprintf("%-30s  %10d  %10d", "Valid posterior draws",
    res_policy.valid_draws, res_policy_change.valid_draws))

# Save
serialize(joinpath(INTER_DIR, "hfi_irf_ratechange.jls"), Dict(
    "irf_point" => res_policy_change.irf,
    "irf_draws" => res_policy_change.irf_draws,
    "H" => H))
println("Results saved to $(joinpath(INTER_DIR, "hfi_irf_ratechange.jls"))")

end  # for s in SAMPLES
