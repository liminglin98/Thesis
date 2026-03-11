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
# Robust Cholesky for covariance-like matrices: symmetrize and add tiny jitter if needed.
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

# Return an SPD matrix close to A by symmetrization + minimal diagonal jitter.
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
##
cd(@__DIR__)
pwd()
##
# Load the data
df = CSV.read("df_final.csv", DataFrame);
##
# Create new variables for positive and negative GDP gaps
df.not_meet_target = df.gdp_gap .< -1;
df.gap_pos = df.gdp_gap .* (1 .- df.not_meet_target);
df.gap_neg = df.gdp_gap .* df.not_meet_target;
##
# Create new variables for exchange rate gap following MANR2026
parse_quarter(s::AbstractString) = begin
    m = match(r"^(\d{4})Q([1-4])$", s)
    m === nothing && return missing
    y = parse(Int, m.captures[1])
    q = parse(Int, m.captures[2])
    Date(y, 3 * (q - 1) + 1, 1)
end

df.quarter_date = parse_quarter.(string.(df.quarter))
df.after2006 = coalesce.(df.quarter_date .>= Date(2006, 1, 1), false)
df.dCNYUSDCPR = df.CNYUSDCPR .- lag(df.CNYUSDCPR, 1)
df.fx_gap = df.dCNYUSDCPR .* df.after2006 .* (df.CNYUSDSpot .- df.CNYUSDCPR)
##
# Create lagged variables for the regression
df.cmpi_l1   = lag(df.cmpi, 1);
df.cpi_gap_l1 = lag(df.cpi_gap, 1);
df.gap_pos_l1 = lag(df.gap_pos, 1);
df.gap_neg_l1 = lag(df.gap_neg, 1);
df.fx_gap_l1 = lag(df.fx_gap, 1);
##
# Drop missing values
df = dropmissing(df, [:cmpi_l1, :cmpi, :cpi_gap_l1, :gap_pos_l1, :gap_neg_l1, :fx_gap_l1])
## 
# Run the regression strictly followig MANR2026
model1 = lm(@formula( cmpi ~ cmpi_l1 + cpi_gap_l1 + gap_pos_l1 + gap_neg_l1 + fx_gap_l1), df)
# Generate predictions from the model
df.pred_cmpi = predict(model1)
df.residuals = residuals(model1)

# Plot residuals from model1
plot(df.quarter, df.residuals,
    label="Model1 residuals",
    legend=:topleft,
    xlabel="Quarter",
    ylabel="Residual",
    title="Model1 Residuals Over Time")
hline!([0.0], linestyle=:dash, color=:black, alpha=0.6, label="Zero line")
## 
# Plot actual vs predicted values
plot(df.quarter, df.cmpi, label="CMPI", legend=:topleft)
plot!(df.quarter, df.pred_cmpi, label="Predicted CMPI", linestyle=:dash)
xlabel!("Quarter")
ylabel!("CMPI")
title!("Actual vs Predicted CMPI")
savefig("CMPI Comparison (MANR2026).png")
## No Lag
df2 = dropmissing(df, [:quarter, :cmpi, :cmpi_l1, :cpi_gap, :gap_pos, :gap_neg, :fx_gap])
model2 = lm(@formula( cmpi ~ cmpi_l1 +cpi_gap + gap_pos + gap_neg + fx_gap), df2)
df2.pred_cmpi2 = predict(model2)
df2.residuals2 = residuals(model2)
plot(df2.quarter, df2.residuals2,
    label="Model2 residuals",
    legend=:topleft,
    xlabel="Quarter",
    ylabel="Residual",
    title=" Residuals (With no lags) Over Time")
hline!([0.0], linestyle=:dash, color=:black, alpha=0.6, label="Zero line")
# Plot actual vs predicted values for model2
plot(df2.quarter, df2.cmpi, label="CMPI", legend=:topleft)
plot!(df2.quarter, df2.pred_cmpi2, label="Predicted CMPI", linestyle=:dash)
xlabel!("Quarter")
ylabel!("CMPI")
title!("Actual vs Predicted CMPI (No Lags)")     
savefig("CMPI Comparison (No Lags).png")
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
