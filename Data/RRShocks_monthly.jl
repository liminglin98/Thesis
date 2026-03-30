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
##
cd(@__DIR__)
pwd()
##
# Load the data
df = CSV.read("romer_china_data.csv", DataFrame);
##
# Calculate monthly GDP growth and CPI gap
df.gdp_gap = df.realgdp_monthly_yoy - df.target_gdp
df.cpi_gap = df.cpi - df.target_cpi
##
# Create new variables for positive and negative GDP gaps
df.not_meet_target = df.gdp_gap .< 1;
df.gap_pos = df.gdp_gap .* (1 .- df.not_meet_target);
df.gap_neg = df.gdp_gap .* df.not_meet_target;
##
# Create new variables for exchange rate gap following MANR2026
df.date = Date.(df.date)
df.after2006 = coalesce.(df.date .>= Date(2006, 1, 1), false)
df.dCNYUSDCPR = df.CNYUSDCPR .- lag(df.CNYUSDCPR, 1)
df.fx_gap = df.dCNYUSDCPR .* df.after2006 .* (df.CNYUSDSpot .- df.CNYUSDCPR)
##
# Create lagged variables for the regression
df.FR007_l1   = lag(df.FR007, 1);
df.cpi_gap_l1 = lag(df.cpi_gap, 1);
df.gap_pos_l1 = lag(df.gap_pos, 1);
df.gap_neg_l1 = lag(df.gap_neg, 1);
df.fx_gap_l1 = lag(df.fx_gap, 1);
## 
# Run the regression strictly followig MANR2026
cols = [:FR007, :FR007_l1, :cpi_gap_l1, :gap_pos_l1, :gap_neg_l1, :fx_gap_l1]
complete = completecases(df[:, cols])

model1 = lm(@formula(FR007 ~ FR007_l1 + cpi_gap_l1 + gap_pos_l1 + gap_neg_l1 + fx_gap_l1), df[complete, :])

df.pred_FR007 = Vector{Union{Missing, Float64}}(missing, nrow(df))
df.residuals  = Vector{Union{Missing, Float64}}(missing, nrow(df))
df.pred_FR007[complete] = predict(model1)
df.residuals[complete]  = residuals(model1)

##
# Plot residuals from model1
plot(df.date, df.residuals,
    label="Model1 residuals",
    legend=:topleft,
    xlabel="Date",
    ylabel="Residual",
    title="FR007 Residuals Over Time")
hline!([0.0], linestyle=:dash, color=:black, alpha=0.6, label="Zero line")
## 
# Plot actual vs predicted values
plot(df.date, df.FR007, label="Actual FR007", legend=:topleft)
plot!(df.date, df.pred_FR007, label="Predicted FR007", linestyle=:dash)
xlabel!("Date")
ylabel!("FR007")
title!("Actual vs Predicted FR007")
##
# Run the exact same regression excluding 2020
df_ex2020 = df[year.(df.date) .!= 2020, :]
cols_ex = [:FR007, :FR007_l1, :cpi_gap_l1, :gap_pos_l1, :gap_neg_l1, :fx_gap_l1]
complete_ex = completecases(df_ex2020[:, cols_ex])

model_ex2020 = lm(@formula(FR007 ~ FR007_l1 + cpi_gap_l1 + gap_pos_l1 + gap_neg_l1 + fx_gap_l1), df_ex2020[complete_ex, :])

df_ex2020 = df_ex2020[complete_ex, :]
df_ex2020.pred_FR007_ex = predict(model_ex2020)
df_ex2020.residuals_ex  = residuals(model_ex2020)

println("\n=== Model (full sample) ===")
println(model1)
println("\n=== Model (excluding 2020) ===")
println(model_ex2020)
##
# Plot residuals: full sample vs excluding 2020
p_res = plot(df.date, df.residuals,
    label="Full sample", legend=:topleft,
    xlabel="Date", ylabel="Residual",
    title="FR007 Residuals: Full vs Excl. 2020",
    color=:blue, alpha=0.7)
plot!(p_res, df_ex2020.date, df_ex2020.residuals_ex,
    label="Excl. 2020", color=:red, linestyle=:dash, alpha=0.7)
hline!([0.0], linestyle=:dot, color=:black, alpha=0.5, label="")
display(p_res)
savefig(p_res, "FR007_residuals_comparison.png")
##
# Plot actual vs predicted: full sample vs excluding 2020
p_fit = plot(df.date, df.FR007,
    label="Actual FR007", legend=:topleft,
    xlabel="Date", ylabel="FR007",
    title="Actual vs Predicted FR007: Full vs Excl. 2020",
    color=:black)
plot!(p_fit, df.date, df.pred_FR007,
    label="Predicted (full)", color=:blue, linestyle=:dash)
plot!(p_fit, df_ex2020.date, df_ex2020.pred_FR007_ex,
    label="Predicted (excl. 2020)", color=:red, linestyle=:dot)
display(p_fit)
savefig(p_fit, "FR007_fit_comparison.png")
##
# Plot coefficient comparison
coef_full   = coef(model1)
coef_ex     = coef(model_ex2020)
coef_names  = coefnames(model1)
n_coef      = length(coef_names)

p_coef = plot(1:n_coef, coef_full,
    seriestype=:bar, label="Full sample",
    xticks=(1:n_coef, coef_names), xrotation=30,
    ylabel="Coefficient", title="Coefficient Comparison: Full vs Excl. 2020",
    alpha=0.6, color=:blue, legend=:topright)
plot!(p_coef, (1:n_coef) .+ 0.3, coef_ex,
    seriestype=:bar, label="Excl. 2020",
    alpha=0.6, color=:red, bar_width=0.3)
display(p_coef)
savefig(p_coef, "FR007_coef_comparison.png")



##============================================================================
# BVAR with Minnesota Priors + IV-SVAR Identification
# ============================================================================
#
# Following MANR2026 / Giannone, Lenza & Primiceri (2015)
#
# VAR variables: [realgdp_monthly_yoy, cpi, FR007, CNYUSDSpot, trade_balance, current_consumption]
# Instrument:    RR residuals (external, NOT in the VAR)
# Priors:        Minnesota (Normal-Inverse-Wishart)
# Lags:          6
# Normalization:  +1 pp FR007 on impact
# Zero restriction: GDP does not respond on impact
# ============================================================================

using LinearAlgebra, Random, Distributions, Printf, Statistics

# =========================
# 1) Data preparation
# =========================
# Transform to YoY growth rates (matching MANR2026)
# Skip variables already in growth rates or percentage points
rename!(df, "trade balance" => :trade_balance, "current consumption" => :current_consumption)
df.CNYUSDSpot_yoy = (df.CNYUSDSpot ./ lag(df.CNYUSDSpot, 12) .- 1) .* 100
df.current_consumption_yoy = (df.current_consumption ./ lag(df.current_consumption, 12) .- 1) .* 100

var_syms = [:realgdp_monthly_yoy, :cpi, :FR007, :CNYUSDSpot_yoy, :current_consumption_yoy, :IP_yoy]
all_syms = vcat(var_syms, [:residuals])

df_bvar = dropmissing(df, all_syms)
dates_bvar = df_bvar.date

n = length(var_syms)
Y_full = Matrix(df_bvar[:, var_syms])
z_full = Vector(df_bvar[:, :residuals])
T_raw  = size(Y_full, 1)

p = 6   # lags
H = 24  # IRF horizon (months)

gdp_col    = findfirst(==(:realgdp_monthly_yoy), var_syms)
policy_col = findfirst(==(:FR007), var_syms)

var_labels = ["Real GDP Growth", "CPI", "FR007", "CNY/USD Spot", "Current Consumption", "Industrial Production"]

println("Variables: ", var_syms)
println("GDP column: ", gdp_col, " → ", var_syms[gdp_col])
println("Policy column: ", policy_col, " → ", var_syms[policy_col])

# Build VAR matrices
Teff = T_raw - p
Y_dep = Y_full[p+1:end, :]   # Teff × n

# Regressors: intercept + p lags
X_var = ones(Teff, 1 + n * p)
for L in 1:p
    X_var[:, 1 + (L-1)*n+1 : 1 + L*n] = Y_full[p+1-L:end-L, :]
end

# Instrument aligned to estimation sample
z = z_full[p+1:end]
dates_est = dates_bvar[p+1:end]

k = size(X_var, 2)   # total number of regressors per equation

println("Sample: ", dates_est[1], " — ", dates_est[end])
println("T_eff = ", Teff, ", n = ", n, ", p = ", p, ", k = ", k)

# =========================
# 2) Minnesota prior setup (GLP 2015)
# =========================
#
# The Minnesota prior shrinks VAR coefficients toward:
#   - Own first lag = 1 (random walk for levels, or 0 for growth rates)
#   - All other coefficients = 0
#
# Key hyperparameters:
#   λ  = overall tightness (how much to shrink toward prior)
#   d  = lag decay (higher = faster shrinkage on distant lags)
#
# Since variables are in YoY growth rates (and interest rates in levels),
# we set the prior mean on own first lag to 1 for all variables.
# This captures persistence in YoY transformations.

# --- Hyperparameters ---
λ = 0.2    # overall tightness (GLP2015 typical range: 0.1–0.5)
d = 1.0    # lag decay exponent (harmonic decay: 1/l^d)

# --- Prior on coefficients: B ~ MN(B0, Ω) ---
# B is k × n, where k = 1 + n*p
# B0: prior mean
# Ω: prior variance (diagonal, equation-by-equation)

# OLS residual variances (for scaling)
B_ols = X_var \ Y_dep
U_ols = Y_dep - X_var * B_ols
sigma2_ols = vec(var(U_ols, dims=1))   # n × 1

# Prior mean: intercept = 0, own first lag = 1, everything else = 0
B0 = zeros(k, n)
for j in 1:n
    # Own first lag position: row 1 + (0)*n + j = 1 + j
    B0[1 + j, j] = 1.0
end

# Prior variance for each coefficient
# V[B_{l,ij}] = (λ / l^d)^2 * (σ_j^2 / σ_i^2)  for lag l, eq j, var i
# V[intercept] = large (diffuse)
Omega_diag = zeros(k, n)   # k × n: variance for each (regressor, equation)

for j in 1:n  # equation j
    # Intercept: diffuse
    Omega_diag[1, j] = 100.0 * sigma2_ols[j]
    
    for L in 1:p
        for i in 1:n  # variable i at lag L
            row = 1 + (L-1)*n + i
            Omega_diag[row, j] = (λ / L^d)^2 * (sigma2_ols[j] / sigma2_ols[i])
        end
    end
end

# =========================
# 3) Posterior (Normal-Inverse-Wishart conjugate)
# =========================
#
# With a diagonal prior on B (equation by equation Minnesota):
#   Posterior for equation j:
#     b_j | Σ, Y ~ N(b_j_post, V_j_post)
#
# For Σ we use the residual-based estimate from the posterior mean of B.
#
# This is the "independent Normal-Wishart" approximation commonly used
# for Minnesota priors (as in Kadiyala & Karlsson 1997).

# Posterior for each equation (vectorized)
B_post = zeros(k, n)
V_post = Vector{Matrix{Float64}}(undef, n)

for j in 1:n
    # Prior precision × prior mean
    Omega_j = Diagonal(Omega_diag[:, j])
    Omega_j_inv = Diagonal(1.0 ./ Omega_diag[:, j])
    
    # Posterior precision
    V_j_post_inv = Omega_j_inv + X_var' * X_var
    V_j_post = Symmetric(inv(V_j_post_inv))
    
    # Posterior mean
    b_j_post = V_j_post * (Omega_j_inv * B0[:, j] + X_var' * Y_dep[:, j])
    
    B_post[:, j] = b_j_post
    V_post[j] = Matrix(V_j_post)
end

# Posterior residuals and Sigma
U_post = Y_dep - X_var * B_post
Sigma_post = (U_post' * U_post) / Teff

println("\n" * "="^60)
println("BVAR Posterior Estimates")
println("="^60)
println("Minnesota hyperparameters: λ = ", λ, ", d = ", d)
println("Posterior Σ diagonal: ", [round(Sigma_post[j,j], digits=6) for j in 1:n])

# =========================
# 4) IV identification (point estimate)
# =========================

function iv_identify(U::Matrix{Float64}, z::Vector{Float64},
                     policy_col::Int, gdp_col::Int)
    Teff_loc, n_loc = size(U)
    u_policy = U[:, policy_col]
    
    gamma_hat = (z' * z) \ (z' * u_policy)
    u_hat = z .* gamma_hat
    
    SS_res = sum((u_policy .- u_hat).^2)
    SS_tot = sum((u_policy .- mean(u_policy)).^2)
    R2 = 1.0 - SS_res / SS_tot
    F_stat = R2 / (1.0 - R2) * (Teff_loc - 2)
    
    denom = u_hat' * u_hat
    if denom < 1e-14
        return nothing, 0.0, 0.0
    end
    
    b_rel = zeros(n_loc)
    for j in 1:n_loc
        b_rel[j] = (u_hat' * U[:, j]) / denom
    end
    
    # b_rel[gdp_col] = 0.0
    
    if abs(b_rel[policy_col]) < 1e-10
        return nothing, F_stat, gamma_hat
    end
    b_rel ./= b_rel[policy_col]
    
    return b_rel, F_stat, gamma_hat
end

b_point, F_stat, gamma_hat = iv_identify(U_post, z, policy_col, gdp_col)

if isnothing(b_point)
    error("IV identification failed on posterior mean")
end

println(@sprintf("\nFirst-stage F-stat: %.2f", F_stat))
println(@sprintf("Corr(z, u_policy): %.4f", cor(z, U_post[:, policy_col])))
println("\nImpact vector (+1 pp FR007):")
for j in 1:n
    println(@sprintf("  %-25s  %+.6f", var_labels[j], b_point[j]))
end

# =========================
# 5) Utility functions for IRFs
# =========================

function get_lag_matrices(B::Matrix{Float64}, n::Int, p::Int)
    A = Vector{Matrix{Float64}}(undef, p)
    for L in 1:p
        rows = 1 + (L-1)*n+1 : 1 + L*n
        A[L] = transpose(B[rows, :])
    end
    return A
end

function compute_ma(A::Vector{Matrix{Float64}}, n::Int, p::Int, H::Int)
    C = [zeros(n, n) for _ in 0:H]
    C[1] = Matrix(I, n, n)
    for h in 1:H
        Ch = zeros(n, n)
        for L in 1:min(p, h)
            Ch .+= A[L] * C[h - L + 1]
        end
        C[h + 1] = Ch
    end
    return C
end

function compute_irfs(C::Vector{Matrix{Float64}}, b::Vector{Float64}, H::Int)
    n_loc = length(b)
    irf = zeros(H + 1, n_loc)
    for h in 0:H
        irf[h + 1, :] = C[h + 1] * b
    end
    return irf
end

# Point estimate IRFs
A_post = get_lag_matrices(B_post, n, p)
C_post = compute_ma(A_post, n, p, H)
irf = compute_irfs(C_post, b_point, H)

# =========================
# 6) Posterior simulation (Gibbs-style draws)
# =========================
#
# For each draw:
#   (a) Draw B* from posterior N(B_post_j, V_post_j) for each equation j
#   (b) Compute residuals U* = Y - X * B*
#   (c) Run IV identification on (U*, z)
#   (d) Compute IRFs from the drawn B* (and hence C*)
#
# This properly accounts for:
#   - Parameter uncertainty (Minnesota posterior)
#   - IV estimation uncertainty (through different residual realizations)

n_draws = 5000
irf_draws = zeros(n_draws, H + 1, n)
n_valid = Ref(0)

Random.seed!(42)

for d_iter in 1:n_draws
    # (a) Draw VAR coefficients from posterior
    B_draw = zeros(k, n)
    for j in 1:n
        # Draw from N(b_j_post, V_j_post)
        b_j_mean = B_post[:, j]
        b_j_cov  = V_post[j]
        
        # Cholesky of posterior covariance
        C_cov = cholesky(Symmetric(b_j_cov + 1e-12 * I(k)); check=false)
        if !issuccess(C_cov)
            # Fallback: eigendecomposition
            E = eigen(Symmetric(b_j_cov))
            vals = max.(E.values, 1e-12)
            sqrtV = E.vectors * Diagonal(sqrt.(vals))
            B_draw[:, j] = b_j_mean + sqrtV * randn(k)
            continue
        end
        B_draw[:, j] = b_j_mean + C_cov.L * randn(k)
    end
    
    # (b) Compute residuals under drawn coefficients
    U_draw = Y_dep - X_var * B_draw
    
    # (c) IV identification
    b_draw, _, _ = iv_identify(U_draw, z, policy_col, gdp_col)
    
    if isnothing(b_draw)
        continue
    end
    
    # (d) Compute MA coefficients and IRFs from drawn B
    A_draw = get_lag_matrices(B_draw, n, p)
    C_draw = compute_ma(A_draw, n, p, H)
    irf_draw = compute_irfs(C_draw, b_draw, H)
    
    # Discard explosive draws
    if any(abs.(irf_draw) .> 500)
        continue
    end
    
    n_valid[] += 1
    irf_draws[n_valid[], :, :] = irf_draw
end

valid_draws = n_valid[]
irf_draws = irf_draws[1:max(valid_draws, 1), :, :]

println(@sprintf("\nPosterior draws: %d / %d valid", valid_draws, n_draws))

# =========================
# 7) Credible sets
# =========================

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

# Dispersion diagnostic
println("\nPosterior dispersion (std at selected horizons):")
println(@sprintf("  %25s  %8s  %8s  %8s  %8s", "Variable", "h=0", "h=6", "h=12", "h=24"))
for j in 1:n
    s0  = valid_draws > 1 ? std(irf_draws[:, 1, j])   : 0.0
    s6  = valid_draws > 1 ? std(irf_draws[:, 7, j])   : 0.0
    s12 = valid_draws > 1 ? std(irf_draws[:, 13, j])  : 0.0
    s24 = valid_draws > 1 ? std(irf_draws[:, 25, j])  : 0.0
    println(@sprintf("  %25s  %8.4f  %8.4f  %8.4f  %8.4f", var_labels[j], s0, s6, s12, s24))
end

# =========================
# 8) Print IRF table
# =========================

println("\n" * "="^60)
println("BVAR + IV-SVAR: Contractionary MP Shock (+1 pp FR007)")
println("Minnesota prior (λ=$(λ), d=$(d)), RR instrument")
println("GDP zero on impact")
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

# =========================
# 9) Plot IRFs (MANR2026 style)
# =========================

p_plots = []

for j in 1:n
    plt = plot(0:H, irf_median[:, j],
        title = var_labels[j],
        label = "Median",
        xlabel = "Months",
        ylabel = "Response",
        legend = :best,
        linewidth = 2.5,
        color = :darkblue)
    
    # 90% credible set (lighter)
    plot!(plt, 0:H, irf_90_lo[:, j],
        fillrange = irf_90_hi[:, j],
        fillalpha = 0.15,
        fillcolor = :steelblue,
        linealpha = 0,
        label = "90% CS")
    
    # 68% credible set (darker)
    plot!(plt, 0:H, irf_68_lo[:, j],
        fillrange = irf_68_hi[:, j],
        fillalpha = 0.35,
        fillcolor = :steelblue,
        linealpha = 0,
        label = "68% CS")
    
    hline!([0], color = :gray, linestyle = :dash, alpha = 0.5, label = "")
    push!(p_plots, plt)
end

n_cols = 3
n_rows = ceil(Int, n / n_cols)
plot_bvar = plot(p_plots...,
    layout = (n_rows, n_cols),
    size = (380 * n_cols, 300 * n_rows),
    plot_title = "BVAR + IV-SVAR: Contractionary MP Shock (+1 pp FR007)\nMinnesota prior, RR instrument, GDP zero on impact")
display(plot_bvar)
savefig(plot_bvar, "irf_bvar_iv_svar.png")

# =========================
# 10) Diagnostics
# =========================

println("\n" * "="^60)
println("Diagnostics")
println("="^60)
println(@sprintf("First-stage F-stat:          %.2f  %s", F_stat,
    F_stat < 10 ? "⚠ WEAK" : "✓ OK"))
println(@sprintf("Corr(z, u_policy):           %.4f", cor(z, U_post[:, policy_col])))
println(@sprintf("Valid posterior draws:        %d / %d", valid_draws, n_draws))
println(@sprintf("Minnesota λ:                 %.2f", λ))
println(@sprintf("Minnesota d:                 %.1f", d))
println("Zero restriction:            GDP = 0 on impact")
println("Normalization:               +1 pp FR007 on impact")
println("Bands:                       Posterior credible sets (not CI)")
##
