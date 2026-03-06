##
using Pkg
using CSV, DataFrames
using Statistics
using GLM
using StatsModels
using Plots
using Dates
using ShiftedArrays, CovarianceMatrices
##
cd(@__DIR__)
pwd()
##
# Load the data
df = CSV.read("df_final.csv", DataFrame);
##
# Create new variables for positive and negative GDP gaps
df.not_meet_target = df.gdp_gap .< 1;
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
# Plot actual vs predicted values
plot(df.quarter, df.cmpi, label="CMPI", legend=:topleft)
plot!(df.quarter, df.pred_cmpi, label="Predicted CMPI", linestyle=:dash)
xlabel!("Quarter")
ylabel!("CMPI")
title!("Actual vs Predicted CMPI (MANR2026)")

##
# Use no lags
df2 = dropmissing(df, [:cmpi, :cmpi_l1,  :cpi_gap, :gap_pos, :gap_neg, :fx_gap])
model2 = lm(@formula( cmpi ~ cmpi_l1 +cpi_gap + gap_pos + gap_neg + fx_gap), df2)
df2.pred_cmpi2 = predict(model2)
df2.residuals2 = residuals(model2)
plot(df2.quarter, df2.cmpi, label="CMPI", legend=:topleft)
plot!(df2.quarter, df2.pred_cmpi2, label="Predicted CMPI (no lags)", linestyle=:dash)
xlabel!("Quarter")
ylabel!("CMPI")
title!("Actual vs Predicted CMPI (no lags)")
##
# IV-SVAR
# ...existing code...
using LinearAlgebra

# =========================
# IV-SVAR (Proxy SVAR)
# Order: [cmpi, cpi, realgdpgrowth, CNYUSDSpot]
# Instrument: residuals (from model1)
# =========================

# 1) Build estimation sample
vars = [:cmpi, :cpi, :realgdpgrowth, :CNYUSDSpot, :residuals]
df_iv = dropmissing(df, vars)

Y = Matrix(df_iv[:, [:cmpi, :cpi, :realgdpgrowth, :CNYUSDSpot]])  # T x n
z = Vector(df_iv.residuals)                                       # T

# 2) Reduced-form VAR(p) by OLS
p = 4
T, n = size(Y)
Teff = T - p

Ydep = Y[p+1:end, :]  # Teff x n

X = ones(Teff, 1 + n*p)  # intercept + lags
for L in 1:p
    X[:, 1 + (L-1)*n + 1 : 1 + L*n] = Y[p+1-L:end-L, :]
end

B = X \ Ydep                 # coefficients
U = Ydep - X * B             # reduced-form residuals, Teff x n

# align instrument to VAR residual sample
zv = z[p+1:end]

# keep rows with valid instrument
keep = .!ismissing.(zv) .& .!isnan.(zv)
U = U[keep, :]
zv = Float64.(zv[keep])

# 3) Proxy identification (single instrument for shock in first variable: cmpi)
u1 = U[:, 1]
u2 = U[:, 2:end]

cov_zu1 = mean((zv .- mean(zv)) .* (u1 .- mean(u1)))
cov_zu2 = vec(mean((zv .- mean(zv)) .* (u2 .- mean(u2)), dims=1))

# impact vector normalized with b1 = 1
b = vcat(1.0, cov_zu2 ./ cov_zu1)

# optional sign normalization: positive policy shock raises cmpi on impact
if b[1] < 0
    b .*= -1
end

println("Impact vector b (unit policy shock): ", b)

# 4) IRFs
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

irf = zeros(H+1, n)
for h in 0:H
    irf[h+1, :] = C[h+1] * b
end

# 5) Plot IRFs
labels = ["cmpi", "cpi", "realgdpgrowth", "CNYUSDSpot"]
p1 = plot(0:H, irf[:,1], title="IRF of cmpi", label="", xlabel="Horizon", ylabel="Response")
p2 = plot(0:H, irf[:,2], title="IRF of cpi", label="", xlabel="Horizon", ylabel="Response")
p3 = plot(0:H, irf[:,3], title="IRF of realgdpgrowth", label="", xlabel="Horizon", ylabel="Response")
p4 = plot(0:H, irf[:,4], title="IRF of CNYUSDSpot", label="", xlabel="Horizon", ylabel="Response")
plot(p1, p2, p3, p4, layout=(2,2), size=(900,600))
# ...existing code...