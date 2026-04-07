##
using CSV, DataFrames
using Statistics
using Plots
using Dates
using Printf
using Random, Distributions
using LinearAlgebra
using Serialization

include(joinpath(@__DIR__, "common.jl"))
const DIAG_DIR = diagnostics_dir()
const INTER_DIR = intermediate_dir()
##
# =============================================================================
# BVAR with Minnesota Prior (Giannone, Lenza & Primiceri 2015)
# 7-variable system — monthly Chinese macro data
# Variables: gdp_yoy, ip_yoy, cpi_yoy, policy_rate, m2_yoy, neer_yoy, usip_yoy
# Lags: p = 6  |  Wold horizon: H = 120
# =============================================================================

# ── 1) Load & prepare data ─────────────────���─────────────────────────────────
df = CSV.read(joinpath(DERIVED_DIR, "china_longterm_data.csv"), DataFrame)
df.date = Date.(df.date)
sort!(df, :date)
##
var_syms  = [:realgdp_monthly_yoy, :IP_yoy, :cpi, :FR007, :M2_growth, :neer_yoy, :US_IP_yoy]
var_labels = ["Real GDP", "IP", "CPI", "FR007", "M2", "NEER", "US IP"]
n = length(var_syms)

df_est = dropmissing(df, var_syms)
df_est = sort(df_est, :date)

T_full  = nrow(df_est)
Y_full  = Matrix{Float64}(df_est[:, var_syms])
dates_full = df_est.date

p = 6    # lags (Miranda-Agrippino et al. 2025)
H = 120  # Wold MA horizon (10 years)

# ── 2) COVID dummies ─────────────────────────────────────────────────────────
covid_dates = [Date(2020,1,1), Date(2020,2,1), Date(2020,3,1), Date(2020,4,1)]
covid_dummy = [d in covid_dates ? 1.0 : 0.0 for d in dates_full]

# ── 3) Build VAR regressor matrix ────────────────────────────────────────────
# X: Teff × (1 + n*p + 4)  [intercept + lags + COVID dummies]
Teff = T_full - p
Y_dep = Y_full[p+1:end, :]            # Teff × n
dates_est = dates_full[p+1:end]

n_covid = length(covid_dates)
k = 1 + n * p + n_covid               # regressors per equation

X = zeros(Teff, k)
X[:, 1] .= 1.0                        # intercept
for L in 1:p
    X[:, 1 + (L-1)*n+1 : 1 + L*n] = Y_full[p+1-L:end-L, :]
end
# COVID dummies aligned to estimation sample (rows p+1 : end)
for (ci, cd) in enumerate(covid_dates)
    col = 1 + n*p + ci
    for t in 1:Teff
        X[t, col] = (dates_est[t] == cd) ? 1.0 : 0.0
    end
end

println("="^60)
println("BVAR Setup")
println("="^60)
println(@sprintf("Variables:  %d", n))
println(@sprintf("Lags:       %d", p))
println(@sprintf("COVID dummies: %s", join(string.(covid_dates), ", ")))
println(@sprintf("Sample:     %s — %s  (Teff = %d)", dates_est[1], dates_est[end], Teff))
println(@sprintf("Regressors per equation: %d", k))

# ── 4) OLS baseline (sanity check) ───────────────────────────────────────────
B_ols  = X \ Y_dep                   # k × n
U_ols  = Y_dep - X * B_ols           # Teff × n
sigma2_ols = vec(var(U_ols, dims=1)) # n

println("\nOLS residual std by variable:")
for j in 1:n
    println(@sprintf("  %-18s  %.4f", var_labels[j], sqrt(sigma2_ols[j])))
end

# ── 5) Minnesota prior ────────────────────────────────────────────────────────
# Hyperparameters (GLP 2015 typical values)
λ = 0.2   # overall tightness
d = 1.0   # lag decay exponent

# Prior mean B0: intercept=0, own first lag=1, all else=0
B0 = zeros(k, n)
for j in 1:n
    B0[1 + j, j] = 1.0   # own first lag → 1
end

# Prior variance Omega_diag[row, eq] for each (regressor, equation)
Omega_diag = zeros(k, n)
for j in 1:n
    Omega_diag[1, j] = 100.0 * sigma2_ols[j]            # intercept: diffuse
    for L in 1:p
        for i in 1:n
            row = 1 + (L-1)*n + i
            Omega_diag[row, j] = (λ / L^d)^2 * (sigma2_ols[j] / sigma2_ols[i])
        end
    end
    # COVID dummies: diffuse
    for ci in 1:n_covid
        Omega_diag[1 + n*p + ci, j] = 100.0 * sigma2_ols[j]
    end
end

# ── 6) Posterior (equation-by-equation Normal conjugate) ─────────────────────
B_post = zeros(k, n)
V_post = Vector{Matrix{Float64}}(undef, n)

XtX = X' * X   # precompute

for j in 1:n
    Omega_j_inv = Diagonal(1.0 ./ Omega_diag[:, j])
    V_j_inv     = Omega_j_inv + XtX
    V_j         = Symmetric(inv(V_j_inv))
    b_j         = V_j * (Omega_j_inv * B0[:, j] + X' * Y_dep[:, j])
    B_post[:, j] = b_j
    V_post[j]    = Matrix(V_j)
end

U_post   = Y_dep - X * B_post
Sigma_u  = (U_post' * U_post) / Teff

println("\nPosterior Σ diagonal (residual variances):")
for j in 1:n
    println(@sprintf("  %-18s  %.6f", var_labels[j], Sigma_u[j,j]))
end

# ── 7) Extract companion-form lag matrices ────────────────────────────────────
A_list = get_lag_matrices(B_post, n, p)
c_vec  = B_post[1, :]   # constant (1 × n → n)

# ── 8) VAR stability check ────────────────────────────────────────────────────
function check_stability(A_list::Vector{Matrix{Float64}})
    p_loc = length(A_list)
    n_loc = size(A_list[1], 1)
    F = zeros(n_loc * p_loc, n_loc * p_loc)
    for L in 1:p_loc
        F[1:n_loc, (L-1)*n_loc+1 : L*n_loc] = A_list[L]
    end
    if p_loc > 1
        F[n_loc+1:end, 1:n_loc*(p_loc-1)] = Matrix(I, n_loc*(p_loc-1), n_loc*(p_loc-1))
    end
    ev = eigvals(F)
    max_mod = maximum(abs.(ev))
    println(@sprintf("\nCompanion matrix max eigenvalue modulus: %.4f  %s",
        max_mod, max_mod >= 1.0 ? "⚠ NOT STABLE" : "✓ Stable"))
    return ev
end

eigenvalues = check_stability(A_list)

# ── 9) Wold MA coefficients ───────────────────────────────────────────────────
function compute_wold(A_list::Vector{Matrix{Float64}}, H::Int)
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

Psi = compute_wold(A_list, H)

# Verify decay: diagonal elements at h = 0, 12, 24, 60, 120
println("\nWold diagonal decay check (Ψ_h[j,j]):")
println(@sprintf("  %-18s  %8s  %8s  %8s  %8s  %8s",
    "Variable", "h=0", "h=12", "h=24", "h=60", "h=120"))
for j in 1:n
    println(@sprintf("  %-18s  %+8.4f  %+8.4f  %+8.4f  %+8.4f  %+8.4f",
        var_labels[j],
        Psi[1][j,j], Psi[13][j,j], Psi[25][j,j], Psi[61][j,j], Psi[121][j,j]))
end

# ── 10) Reduced-form residuals ────────────────────────────────────────────────
function compute_residuals(Y::Matrix{Float64}, A_list, c::Vector{Float64})
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

residuals = compute_residuals(Y_full, A_list, c_vec)

# ── 11) Multi-step forecast from end of sample ────────────────────────────────
H_fc = 24   # forecast horizon for plots

function forecast_from(Y::Matrix{Float64}, A_list, c::Vector{Float64},
                       start_idx::Int, H_fc::Int)
    p_loc = length(A_list)
    history = [Y[start_idx - j + 1, :] for j in 1:p_loc]  # lag 1 first
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

fc = forecast_from(Y_full, A_list, c_vec, T_full, H_fc)
fc_dates = [dates_full[end] + Month(h) for h in 1:H_fc]

# ── 12) Plots ─────────────────────────────────────────────────────────────────
##
# Wold diagonal decay
p_wold = plot(0:H, [Psi[h+1][1,1] for h in 0:H],
    label=var_labels[1], linewidth=1.5,
    xlabel="Horizon (months)", ylabel="Ψ_h[j,j]",
    title="Wold Coefficient Diagonal Decay")
for j in 2:n
    plot!(p_wold, 0:H, [Psi[h+1][j,j] for h in 0:H], label=var_labels[j], linewidth=1.5)
end
hline!([0], color=:black, linestyle=:dash, alpha=0.4, label="")
display(p_wold)
savefig(p_wold, joinpath(DIAG_DIR, "bvar_wold_decay.png"))
##
# Residuals
p_resid_plots = []
for j in 1:n
    plt = plot(dates_est, residuals[:, j],
        label="", color=:steelblue, linewidth=0.8,
        xlabel="Date", ylabel="Residual",
        title=var_labels[j])
    hline!([0], color=:black, linestyle=:dash, alpha=0.5, label="")
    push!(p_resid_plots, plt)
end
n_cols = 3
n_rows = ceil(Int, n / n_cols)
p_resid = plot(p_resid_plots...,
    layout=(n_rows, n_cols),
    size=(380*n_cols, 280*n_rows),
    plot_title="BVAR Reduced-Form Residuals")
display(p_resid)
savefig(p_resid, joinpath(DIAG_DIR, "bvar_residuals.png"))
##
# In-sample fit + forecast: GDP (variable 1)
gdp_idx = 1
p_gdp = plot(dates_est, Y_dep[:, gdp_idx],
    label="Actual", color=:black, linewidth=1.5,
    xlabel="Date", ylabel="% YoY",
    title="GDP YoY: In-Sample Fit & $(H_fc)-Month Forecast")
plot!(p_gdp, dates_est, X * B_post[:, gdp_idx],
    label="Fitted (BVAR)", color=:blue, linestyle=:dash, linewidth=1.5)
plot!(p_gdp, fc_dates, fc[:, gdp_idx],
    label="Forecast", color=:red, linestyle=:dot,
    linewidth=2, marker=:circle, markersize=3)
hline!([0], color=:gray, linestyle=:dot, alpha=0.5, label="")
display(p_gdp)
savefig(p_gdp, joinpath(DIAG_DIR, "bvar_gdp_forecast.png"))
##
# ── 13) Save results ──────────────────────────────────────────────────────────
results = Dict(
    "A_list"         => A_list,
    "c"              => c_vec,
    "Sigma_u"        => Sigma_u,
    "Psi"            => Psi,
    "residuals"      => residuals,
    "variable_names" => var_labels,
    "dates"          => dates_est,
    "p"              => p,
    "H"              => H,
)
serialize(joinpath(INTER_DIR, "bvar_results.jls"), results)
println("\nResults saved to $(joinpath(INTER_DIR, \"bvar_results.jls\"))")
