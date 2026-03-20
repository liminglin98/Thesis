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
##
# IV-SVAR
using LinearAlgebra

if !isdefined(@__MODULE__, :safe_cholesky)
    function safe_cholesky(A::AbstractMatrix{<:Real}; jitter0::Float64 = 1e-10, max_tries::Int = 8)
        A_sym = 0.5 .* (Matrix(A) + Matrix(A)')
        nA = size(A_sym, 1)
        for t in 0:max_tries
            jitter = jitter0 * (10.0^t)
            A_try = A_sym + jitter * I(nA)
            F = cholesky(Symmetric(A_try); check=false)
            if issuccess(F)
                return F
            end
        end
        error("safe_cholesky failed even after jitter escalation")
    end
end

if !isdefined(@__MODULE__, :make_spd)
    function make_spd(A::AbstractMatrix{<:Real}; jitter0::Float64 = 1e-10, max_tries::Int = 8)
        A_sym = 0.5 .* (Matrix(A) + Matrix(A)')
        nA = size(A_sym, 1)
        for t in 0:max_tries
            jitter = jitter0 * (10.0^t)
            A_try = A_sym + jitter * I(nA)
            F = cholesky(Symmetric(A_try); check=false)
            if issuccess(F)
                return Matrix(Symmetric(A_try))
            end
        end
        error("make_spd failed even after jitter escalation")
    end
end

# =========================
# SVAR with Recursive Identification
# Order: [residuals, realgdpgrowth, cpi, cmpi, FR007, CNYUSDSpot]
# residuals = monetary policy shock (from model1)
# =========================

# 1) Build estimation sample
vars = [:residuals, :realgdpgrowth, :cpi, :cmpi, :FR007, :CNYUSDSpot]
df_svar = dropmissing(df, vars)

Y = Matrix(df_svar[:, [:residuals, :realgdpgrowth, :cpi, :cmpi, :FR007, :CNYUSDSpot]])  # T x n
T, n = size(Y)

# 2) Reduced-form VAR(p) by OLS
p = 2  # 2 lags
Teff = T - p

Ydep = Y[p+1:end, :]  # Teff x n

X = ones(Teff, 1 + n*p)  # intercept + lags
for L in 1:p
    X[:, 1 + (L-1)*n + 1 : 1 + L*n] = Y[p+1-L:end-L, :]
end

B = X \ Ydep                 # coefficients
U = Ydep - X * B             # reduced-form residuals, Teff x n

println("\n" * "="^60)
println("SVAR Estimation Results")
println("="^60)
println("Sample size: ", Teff)
println("Variables: [residuals, realgdpgrowth, cpi, cmpi, FR007, CNYUSDSpot]")

# 3) Structural identification via Cholesky decomposition (recursive ordering)
Sigma = (U' * U) / Teff      # variance-covariance matrix
P = safe_cholesky(Sigma).L   # Cholesky decomposition: Sigma = P * P'

# Impact matrix: each column is the impact of one structural shock
# First column = impact of monetary policy shock (residuals shock)
println("\nImpact matrix P (columns = structural shocks):")
display(P)

# 4) IRFs for monetary policy shock (first structural shock)
# Extract A1..Ap from OLS coefficients
A = Vector{Matrix{Float64}}(undef, p)
for L in 1:p
    rows = 1 + (L-1)*n + 1 : 1 + L*n
    A[L] = transpose(B[rows, :])  # n x n
end

H = 10  # horizons
C = [zeros(n, n) for _ in 0:H]
C[1] = Matrix(I, n, n)  # C_0

for h in 1:H
    Ch = zeros(n, n)
    for L in 1:min(p, h)
        Ch .+= A[L] * C[h - L + 1]
    end
    C[h + 1] = Ch
end

# IRF to monetary policy shock (first column of P)
b = P[:, 1]
cmpi_idx = 4
if abs(b[cmpi_idx]) > 1e-10
    # Scale shock so CMPI rises by exactly +1 on impact (t=0)
    b = b ./ b[cmpi_idx]
else
    @warn "CMPI impact is near zero; cannot normalize shock to +1 CMPI at t=0"
end
irf = zeros(H+1, n)
for h in 0:H
    irf[h+1, :] = C[h+1] * b
end

# 5) Bootstrap confidence bands (68%)
n_boot = 1000
irf_boot = zeros(n_boot, H+1, n)
valid_boot = 0

for boot_iter in 1:n_boot
    # Resample with replacement
    boot_idx = rand(1:Teff, Teff)
    U_boot = U[boot_idx, :]
    
    # Re-estimate variance-covariance matrix and Cholesky
    Sigma_boot = (U_boot' * U_boot) / Teff
    P_boot = safe_cholesky(Sigma_boot).L
    b_boot = P_boot[:, 1]
    if abs(b_boot[cmpi_idx]) > 1e-10
        b_boot = b_boot ./ b_boot[cmpi_idx]
    else
        continue
    end
    valid_boot += 1
    
    # Compute IRFs for this bootstrap sample
    for h in 0:H
        irf_boot[valid_boot, h+1, :] = C[h+1] * b_boot
    end
end

if valid_boot == 0
    error("No valid bootstrap draws after CMPI normalization")
end

irf_boot = irf_boot[1:valid_boot, :, :]

# Compute 68% confidence bands (16th and 84th percentiles)
irf_lower = zeros(H+1, n)
irf_upper = zeros(H+1, n)

for h in 0:H
    for j in 1:n
        irf_lower[h+1, j] = quantile(irf_boot[:, h+1, j], 0.16)
        irf_upper[h+1, j] = quantile(irf_boot[:, h+1, j], 0.84)
    end
end

println("\n" * "="^60)
println("SVAR IRF Results (2 lags, with 68% CI)")
println("="^60)
println("Response to monetary policy shock normalized to +1 CMPI at t=0")

# 6) Plot IRFs with 68% confidence bands
labels = ["Residuals", "Real GDP Growth", "CPI", "CMPI", "FR007", "CNY/USD Spot"]
p_plots_svar = []

for j in 1:n
    p = plot(0:H, irf[:,j], 
             title="IRF of $(labels[j])", 
             label="Point Estimate", 
             xlabel="Quarters", 
             ylabel="Response",
             legend=:topright,
             marker=:circle,
             markersize=3,
             linewidth=2,
             color=:green)
    plot!(p, 0:H, irf_lower[:, j],
          fillrange=irf_upper[:, j],
          fillalpha=0.2,
          fillcolor=:green,
          linealpha=0,
          label="68% CI")
    hline!([0], color=:gray, linestyle=:dash, alpha=0.5, label="")
    push!(p_plots_svar, p)
end

plot_svar = plot(p_plots_svar..., layout=(2,3), size=(1000,600))
display(plot_svar)
savefig(plot_svar, "irf_svar.png")
