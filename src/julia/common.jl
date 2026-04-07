# =============================================================================
# Shared utility functions for BVAR/SVAR analysis
# =============================================================================

using LinearAlgebra

# --- Path configuration ---
const PROJECT_ROOT = dirname(dirname(@__DIR__))
const DERIVED_DIR  = joinpath(PROJECT_ROOT, "data", "derived")

function output_dir(subfolder::String)
    d = joinpath(PROJECT_ROOT, "outputs", subfolder)
    mkpath(d)
    return d
end

# --- Categorized output directories ---
function intermediate_dir()
    d = joinpath(PROJECT_ROOT, "outputs", "intermediate")
    mkpath(d)
    return d
end

function main_results_dir()
    d = joinpath(PROJECT_ROOT, "outputs", "main_results")
    mkpath(d)
    return d
end

function diagnostics_dir()
    d = joinpath(PROJECT_ROOT, "outputs", "diagnostics")
    mkpath(d)
    return d
end

function robustness_dir()
    d = joinpath(PROJECT_ROOT, "outputs", "robustness")
    mkpath(d)
    return d
end

# --- VAR lag matrices ---

function get_lag_matrices(B::Matrix{Float64}, n::Int, p::Int)
    A = Vector{Matrix{Float64}}(undef, p)
    for L in 1:p
        rows = 1 + (L-1)*n+1 : 1 + L*n
        A[L] = transpose(B[rows, :])
    end
    return A
end

# --- Wold / MA coefficient computation ---

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

# --- IRF computation ---

function compute_irfs(C::Vector{Matrix{Float64}}, b::Vector{Float64}, H::Int)
    n_loc = length(b)
    irf = zeros(H + 1, n_loc)
    for h in 0:H
        irf[h + 1, :] = C[h + 1] * b
    end
    return irf
end

# --- IV identification ---

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

    if abs(b_rel[policy_col]) < 1e-10
        return nothing, F_stat, gamma_hat
    end
    b_rel ./= b_rel[policy_col]

    return b_rel, F_stat, gamma_hat
end

# --- Robust Cholesky ---

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
