##
using Pkg
using CSV, DataFrames
using Statistics
using GLM
using StatsModels
using Plots
using Dates
using ShiftedArrays, CovarianceMatrices
using Printf
using Random, Distributions
using LinearAlgebra
using Serialization
##
include(joinpath(@__DIR__, "common.jl"))
##
# Load and prepare data once (lags computed on full sample)
df_raw = CSV.read(joinpath(DERIVED_DIR, "romer_china_data.csv"), DataFrame)
df_raw.date = Date.(df_raw.date)

df_raw.gdp_gap = df_raw.realgdp_monthly_yoy - df_raw.target_gdp
df_raw.cpi_gap = df_raw.cpi - df_raw.target_cpi

df_raw.not_meet_target = df_raw.gdp_gap .< 1
df_raw.gap_pos = df_raw.gdp_gap .* (1 .- df_raw.not_meet_target)
df_raw.gap_neg = df_raw.gdp_gap .* df_raw.not_meet_target

df_raw.after2006 = coalesce.(df_raw.date .>= Date(2006, 1, 1), false)
df_raw.dCNYUSDCPR = df_raw.CNYUSDCPR .- lag(df_raw.CNYUSDCPR, 1)
df_raw.fx_gap = df_raw.dCNYUSDCPR .* df_raw.after2006 .* (df_raw.CNYUSDSpot .- df_raw.CNYUSDCPR)

df_raw.FR007_l1   = lag(df_raw.FR007, 1)
df_raw.cpi_gap_l1 = lag(df_raw.cpi_gap, 1)
df_raw.gap_pos_l1 = lag(df_raw.gap_pos, 1)
df_raw.gap_neg_l1 = lag(df_raw.gap_neg, 1)
df_raw.fx_gap_l1  = lag(df_raw.fx_gap, 1)

df_raw.CNYUSDSpot_yoy = (df_raw.CNYUSDSpot ./ lag(df_raw.CNYUSDSpot, 12) .- 1) .* 100

##
# ── Loop over sample periods ────────────────────────────────────────────────

shock_samples = vcat(SAMPLES, SHOCK_SEGMENTS)

for s in shock_samples

println("\n", "="^70)
println("  RRShocks — Sample: $(s.start_date) to $(s.end_date)  [$(s.label)]")
println("="^70)

is_segment = startswith(s.label, "seg_")

ROBUST_DIR = robustness_dir(s.label)
MAIN_DIR   = main_results_dir(s.label)
DIAG_DIR   = diagnostics_dir(s.label)
INTER_DIR  = intermediate_dir(s.label)

fit_out_dir = is_segment ? DIAG_DIR : ROBUST_DIR
irf_out_dir = is_segment ? DIAG_DIR : MAIN_DIR

# Filter by date range
df = filter(r -> !ismissing(r.date) && r.date >= s.start_date && r.date <= s.end_date, df_raw)

# =========================
# 1) Policy rule regression
# =========================
cols = [:FR007, :FR007_l1, :cpi_gap_l1, :gap_pos_l1, :gap_neg_l1, :fx_gap_l1]
complete = completecases(df[:, cols])

model1 = lm(@formula(FR007 ~ FR007_l1 + cpi_gap_l1 + gap_pos_l1 + gap_neg_l1 + fx_gap_l1), df[complete, :])

df.pred_FR007 = Vector{Union{Missing, Float64}}(missing, nrow(df))
df.policy_residual  = Vector{Union{Missing, Float64}}(missing, nrow(df))
df.pred_FR007[complete] = predict(model1)
df.policy_residual[complete]  = GLM.residuals(model1)

println("\n=== Model (full sample $(s.label)) ===")
println(model1)

# =========================
# 2) Policy-rule fit plots (full sample only)
# =========================
if !is_segment
    p_res = plot(df.date, df.policy_residual,
        label="Residual", legend=:topleft,
        xlabel="Date", ylabel="Residual",
        title="FR007 Residuals ($(s.label))",
        color=:blue, alpha=0.8)
    hline!([0.0], linestyle=:dot, color=:black, alpha=0.5, label="")
    display(p_res)
    savefig(p_res, joinpath(fit_out_dir, "FR007_residuals_comparison.png"))

    p_fit = plot(df.date, df.FR007,
        label="Actual FR007", legend=:topleft,
        xlabel="Date", ylabel="FR007",
        title="Actual vs Predicted FR007 ($(s.label))",
        color=:black)
    plot!(p_fit, df.date, df.pred_FR007,
        label="Predicted", color=:blue, linestyle=:dash)
    display(p_fit)
    savefig(p_fit, joinpath(fit_out_dir, "FR007_fit_comparison.png"))
else
    println("Segment mode: skipping policy-rule fit plots; generating IRFs only.")
end

# ============================================================================
# 3) BVAR with Minnesota Priors + IV-SVAR Identification
# ============================================================================

var_syms = [:realgdp_monthly_yoy, :cpi, :FR007, :neer_yoy, :IP_yoy]
all_syms = vcat(var_syms, [:policy_residual])

df_bvar = dropmissing(df, all_syms)
dates_bvar = df_bvar.date

n = length(var_syms)
Y_full = Matrix(df_bvar[:, var_syms])
z_full = Vector(df_bvar[:, :policy_residual])
T_raw  = size(Y_full, 1)

p = 6
H = 24

gdp_col    = findfirst(==(:realgdp_monthly_yoy), var_syms)
policy_col = findfirst(==(:FR007), var_syms)

var_labels = ["Real GDP Growth", "CPI", "FR007", "Real Effective Exchange Rate", "Industrial Value Added"]

Teff = T_raw - p
Y_dep = Y_full[p+1:end, :]

X_var = ones(Teff, 1 + n * p)
for L in 1:p
    X_var[:, 1 + (L-1)*n+1 : 1 + L*n] = Y_full[p+1-L:end-L, :]
end

z = z_full[p+1:end]
dates_est = dates_bvar[p+1:end]
k = size(X_var, 2)

println("\nBVAR Sample: ", dates_est[1], " — ", dates_est[end])
println("T_eff = ", Teff, ", n = ", n, ", p = ", p, ", k = ", k)

# Minnesota prior
λ = 0.2
d = 1.0

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
    for L in 1:p
        for i in 1:n
            row = 1 + (L-1)*n + i
            Omega_diag[row, j] = (λ / L^d)^2 * (sigma2_ols[j] / sigma2_ols[i])
        end
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
Sigma_post = (U_post' * U_post) / Teff

println("\nBVAR Posterior Estimates")
println("Minnesota hyperparameters: λ = ", λ, ", d = ", d)

# IV identification (point estimate)
b_point, F_stat, gamma_hat = iv_identify(U_post, z, policy_col, gdp_col)

if isnothing(b_point)
    error("IV identification failed on posterior mean")
end

println(@sprintf("\nFirst-stage F-stat: %.2f", F_stat))
println(@sprintf("Corr(z, u_policy): %.4f", cor(z, U_post[:, policy_col])))

# Point estimate IRFs
A_post = get_lag_matrices(B_post, n, p)
C_post = compute_ma(A_post, n, p, H)
irf = compute_irfs(C_post, b_point, H)

# Posterior simulation
n_draws = 5000
irf_draws = zeros(n_draws, H + 1, n)
n_valid = Ref(0)

Random.seed!(42)

for d_iter in 1:n_draws
    B_draw = zeros(k, n)
    for j in 1:n
        b_j_mean = B_post[:, j]
        b_j_cov  = V_post[j]
        C_cov = cholesky(Symmetric(b_j_cov + 1e-12 * I(k)); check=false)
        if !issuccess(C_cov)
            E = eigen(Symmetric(b_j_cov))
            vals = max.(E.values, 1e-12)
            sqrtV = E.vectors * Diagonal(sqrt.(vals))
            B_draw[:, j] = b_j_mean + sqrtV * randn(k)
            continue
        end
        B_draw[:, j] = b_j_mean + C_cov.L * randn(k)
    end

    U_draw = Y_dep - X_var * B_draw
    b_draw, _, _ = iv_identify(U_draw, z, policy_col, gdp_col)
    isnothing(b_draw) && continue

    A_draw = get_lag_matrices(B_draw, n, p)
    C_draw = compute_ma(A_draw, n, p, H)
    irf_draw = compute_irfs(C_draw, b_draw, H)

    if any(abs.(irf_draw) .> 500)
        continue
    end

    n_valid[] += 1
    irf_draws[n_valid[], :, :] = irf_draw
end

valid_draws = n_valid[]
irf_draws = irf_draws[1:max(valid_draws, 1), :, :]

println(@sprintf("\nPosterior draws: %d / %d valid", valid_draws, n_draws))

# Credible sets
irf_68_lo = zeros(H + 1, n)
irf_68_hi = zeros(H + 1, n)
irf_90_lo = zeros(H + 1, n)
irf_90_hi = zeros(H + 1, n)
irf_median = zeros(H + 1, n)

if valid_draws > 1
    for h in 0:H
        for j in 1:n
            draws = irf_draws[:, h+1, j]
            irf_median[h+1, j] = quantile(draws, 0.50)
            irf_68_lo[h+1, j]  = quantile(draws, 0.16)
            irf_68_hi[h+1, j]  = quantile(draws, 0.84)
            irf_90_lo[h+1, j]  = quantile(draws, 0.05)
            irf_90_hi[h+1, j]  = quantile(draws, 0.95)
        end
    end
end

# IRF table
println("\n" * "="^60)
println("BVAR + IV-SVAR: Contractionary MP Shock (+1 pp FR007) [$(s.label)]")
println("="^60)

for j in 1:n
    println("\n--- $(var_labels[j]) ---")
    println(@sprintf("  %4s  %8s  %8s  [%8s, %8s]  [%8s, %8s]",
        "h", "Median", "Point", "68lo", "68hi", "90lo", "90hi"))
    for h in [0, 1, 3, 6, 9, 12, 18, 24]
        if h <= H
            println(@sprintf("  %4d  %+8.4f  %+8.4f  [%+8.4f, %+8.4f]  [%+8.4f, %+8.4f]",
                h, irf_median[h+1, j], irf[h+1, j],
                irf_68_lo[h+1, j], irf_68_hi[h+1, j],
                irf_90_lo[h+1, j], irf_90_hi[h+1, j]))
        end
    end
end

# IRF plot
p_plots = []
for j in 1:n
    plt = plot(0:H, irf_median[:, j],
        title = var_labels[j],
        label = "Median",
        xlabel = "Months", ylabel = "Response",
        legend = :best, linewidth = 2.5, color = :darkblue)
    plot!(plt, 0:H, irf_68_lo[:, j],
        fillrange = irf_68_hi[:, j],
        fillalpha = 0.35, fillcolor = :steelblue, linealpha = 0, label = "68% CS")
    hline!([0], color = :gray, linestyle = :dash, alpha = 0.5, label = "")
    push!(p_plots, plt)
end

n_cols = 3
n_rows = ceil(Int, n / n_cols)
plot_bvar = plot(p_plots...,
    layout = (n_rows, n_cols),
    size = (380 * n_cols, 300 * n_rows),
    plot_title = "BVAR + IV-SVAR: +1 pp FR007 shock — RR instrument ($(s.label))")
display(plot_bvar)
savefig(plot_bvar, joinpath(irf_out_dir, "irf_bvar_iv_svar.png"))

# Diagnostics
println("\n" * "="^60)
println("Diagnostics [$(s.label)]")
println("="^60)
println(@sprintf("First-stage F-stat:          %.2f  %s", F_stat,
    F_stat < 10 ? "⚠ WEAK" : "✓ OK"))
println(@sprintf("Valid posterior draws:        %d / %d", valid_draws, n_draws))

# Save
serialize(joinpath(INTER_DIR, "narrative_irf_results.jls"), Dict(
    "irf_point" => irf, "irf_draws" => irf_draws, "H" => H))

println("Results saved to $(joinpath(INTER_DIR, "narrative_irf_results.jls"))")

end  # for s in SAMPLES
