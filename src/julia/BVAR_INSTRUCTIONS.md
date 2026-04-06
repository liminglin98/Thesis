# BVAR Estimation: Instructions for Claude Code
## Task: Estimate reduced-form BVAR and extract Wold representation

---

## What This Step Does

We estimate a 7-variable Bayesian VAR on monthly Chinese macro data. The VAR is a
pure forecasting model — no structural identification. From it we extract two objects
needed for the counterfactual exercise downstream:

1. **Wold MA coefficients** Ψ_0, Ψ_1, ..., Ψ_H — the propagation structure of all shocks
2. **Reduced-form residuals** u_t — the history of forecast surprises at each date

---

## Input

A CSV file `var_dataset.csv` with 8 columns:

| Column | Variable | Units | Transformation |
|--------|----------|-------|----------------|
| `date` | Monthly date | datetime | YYYY-MM-DD |
| `gdp_yoy` | Monthly real GDP growth | % | YoY growth rate (from Stock-Watson interpolation) |
| `ip_yoy` | Chinese industrial production growth | % | YoY growth rate (released this way by NBS) |
| `cpi_yoy` | CPI inflation | % | YoY growth rate |
| `policy_rate` | 7-day reverse repo rate | % | Level (already in pct points, NO transformation) |
| `m2_yoy` | M2 money supply growth | % | YoY growth rate |
| `neer_yoy` | RMB nominal effective exchange rate growth | % | YoY growth rate |
| `usip_yoy` | US industrial production growth | % | YoY growth rate (FRED: INDPRO) |

- Sample: 2003:01 – 2025:12 (276 rows)
- All variables should be stationary after these transformations
- There may be NaNs at the start (first 12 months for YoY variables); the effective
  estimation sample starts after accounting for both YoY lags and VAR lags

---

## BVAR Specification

### Model

$$y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_6 y_{t-6} + u_t$$

where y_t is a 7×1 vector, c is a 7×1 constant, each A_j is 7×7, and u_t ~ N(0, Σ)

### Settings
- **Variables (n=7)**: gdp_yoy, ip_yoy, cpi_yoy, policy_rate, m2_yoy, neer_yoy, usip_yoy
- **Lags (p=6)**: consistent with Miranda-Agrippino et al. (2025)
- **Deterministics**: constant + COVID dummy variables
- **COVID dummies**: at minimum, indicator variables for 2020:02, 2020:03, 2020:04
  (set to 1 in those months, 0 otherwise). Consider also 2020:01 if the data shows
  an outlier there.

### Minnesota Prior (Giannone, Lenza, Primiceri 2015)

The Minnesota prior shrinks VAR coefficients toward a "naive" benchmark:
- Own first lag ≈ 1 (random walk), all other coefficients ≈ 0
- **IMPORTANT**: since all our variables except the policy rate are YoY growth rates
  (which are stationary/mean-reverting, not unit root), the prior on own first lag
  should arguably be set closer to 0 or to a moderate persistence value (e.g., 0.5–0.8)
  rather than 1. This is a setting to check in whatever BVAR package is used.
  Alternatively, use the standard unit-root prior and let the data dominate — with
  276 observations the prior won't matter much.

Key hyperparameters:
- λ (overall tightness): controls how much coefficients can deviate from prior mean.
  Giannone-Lenza-Primiceri recommend selecting via marginal likelihood maximization.
- Cross-variable shrinkage: lags of OTHER variables shrink more than own lags

### Implementation Options (Python)

**Option A: `bvartools` or `bayesianVARs`** (if available)
- These R packages are well-documented; could call via rpy2

**Option B: Custom implementation**
- The Minnesota prior with known Σ reduces to equation-by-equation OLS with a
  modified (augmented) regression. This is straightforward to implement:
  1. Stack the VAR as Y = X B + U (where Y is T×n, X is T×(np+1), B is (np+1)×n)
  2. The Minnesota prior on vec(B) is Normal with known mean and precision
  3. Posterior mean = (X'X + Ω_prior⁻¹)⁻¹ (X'Y + Ω_prior⁻¹ B_prior)
  4. This is a standard linear regression with conjugate prior

**Option C: `statsmodels` VAR + manual prior implementation**
- Estimate frequentist VAR first as a baseline/sanity check
- Then add the prior manually

**For a working baseline**: start with a simple frequentist OLS VAR via statsmodels
to verify the data pipeline and Wold computation work correctly. Then add the
Minnesota prior as refinement.

---

## Outputs to Compute

### Output 1: VAR Coefficient Matrices

```python
# After estimation, store:
A = [A_1, A_2, ..., A_6]  # list of 7×7 numpy arrays
c = ...                      # 7×1 constant vector
Sigma_u = ...                # 7×7 residual covariance matrix
```

### Output 2: Wold MA Coefficients

Compute recursively from the A matrices:

```python
def compute_wold_coefficients(A_list, H=120):
    """
    Compute Wold MA coefficients Ψ_0, Ψ_1, ..., Ψ_H from VAR coefficient
    matrices A_1, ..., A_p.

    Parameters
    ----------
    A_list : list of np.ndarray
        VAR coefficient matrices [A_1, A_2, ..., A_p], each shape (n, n)
    H : int
        Maximum horizon (e.g., 120 months = 10 years)

    Returns
    -------
    Psi : list of np.ndarray
        Wold coefficients [Ψ_0, Ψ_1, ..., Ψ_H], each shape (n, n)
    """
    p = len(A_list)
    n = A_list[0].shape[0]

    Psi = [np.eye(n)]  # Ψ_0 = I

    for h in range(1, H + 1):
        Psi_h = np.zeros((n, n))
        for j in range(1, min(h, p) + 1):
            Psi_h += A_list[j - 1] @ Psi[h - j]
        Psi.append(Psi_h)

    return Psi
```

**Verification**: Ψ_h should decay toward zero as h grows. Plot the (i,i) diagonal
elements of Ψ_h for each variable — they should converge to 0. If they don't,
there may be a unit root issue or estimation problem.

### Output 3: Reduced-Form Residuals

```python
def compute_residuals(Y, A_list, c):
    """
    Compute 1-step-ahead forecast errors u_t = y_t - ŷ_{t|t-1}

    Parameters
    ----------
    Y : np.ndarray, shape (T, n)
        Full data matrix (rows = time, columns = variables)
    A_list : list of np.ndarray
        [A_1, ..., A_p]
    c : np.ndarray, shape (n,)
        Constant vector

    Returns
    -------
    residuals : np.ndarray, shape (T - p, n)
        Reduced-form residuals starting from period p+1
    """
    T, n = Y.shape
    p = len(A_list)
    residuals = np.zeros((T - p, n))

    for t in range(p, T):
        y_hat = c.copy()
        for j in range(p):
            y_hat += A_list[j] @ Y[t - 1 - j]
        residuals[t - p] = Y[t] - y_hat

    return residuals
```

### Output 4: Multi-Step Forecasts

For the conditional counterfactual, we need forecasts from specific dates:

```python
def forecast_from_date(Y, A_list, c, start_idx, H=60):
    """
    Produce H-step-ahead iterated forecasts starting from date start_idx.

    Parameters
    ----------
    Y : np.ndarray, shape (T, n)
        Full data matrix
    A_list : list of np.ndarray
        [A_1, ..., A_p]
    c : np.ndarray, shape (n,)
        Constant vector
    start_idx : int
        Index in Y from which to start forecasting (uses Y[start_idx-p:start_idx]
        as initial conditions)
    H : int
        Forecast horizon in months

    Returns
    -------
    forecasts : np.ndarray, shape (H, n)
        Forecasted values ŷ_{start+1}, ..., ŷ_{start+H}
    """
    p = len(A_list)
    n = A_list[0].shape[0]

    # Build initial history buffer
    history = list(Y[start_idx - p:start_idx])  # last p observations
    forecasts = []

    for h in range(H):
        y_hat = c.copy()
        for j in range(p):
            y_hat += A_list[j] @ history[-(j + 1)]
        forecasts.append(y_hat)
        history.append(y_hat)

    return np.array(forecasts)
```

---

## Diagnostic Checks

### 1. Residual Properties
- Plot residual time series for each variable — should look like white noise
  (except possibly around COVID)
- Check residual autocorrelation: Ljung-Box test or plot ACF of residuals
- Residuals during COVID dummy months may still be large; that's OK

### 2. Forecast Accuracy
- Pseudo out-of-sample exercise: estimate VAR on 2003–2015, forecast 2016–2019,
  compare to actuals
- Report RMSE for GDP growth, CPI, and policy rate (the three "core" variables)
- Wolf et al. compare their VAR forecasts to Survey of Professional Forecasters;
  we don't have a direct Chinese equivalent, but can compare to IMF WEO or
  Bloomberg consensus if available

### 3. Wold Coefficient Decay
- Plot diagonal elements of Ψ_h for h = 0, 1, ..., 120
- All should converge to zero
- If GDP or IP coefficients are very persistent (slow decay), that's expected for
  real activity; but if they don't converge, check for unit roots

### 4. Stability
- Check VAR stability: all eigenvalues of the companion matrix should be inside
  the unit circle
- The companion matrix is the (7p × 7p) matrix:

```
F = | A_1  A_2  A_3  A_4  A_5  A_6 |
    | I    0    0    0    0    0    |
    | 0    I    0    0    0    0    |
    | 0    0    I    0    0    0    |
    | 0    0    0    I    0    0    |
    | 0    0    0    0    I    0    |
```

```python
def check_stability(A_list):
    """Check all eigenvalues of companion matrix are inside unit circle."""
    p = len(A_list)
    n = A_list[0].shape[0]
    companion = np.zeros((n * p, n * p))
    for j in range(p):
        companion[:n, j * n:(j + 1) * n] = A_list[j]
    if p > 1:
        companion[n:, :n * (p - 1)] = np.eye(n * (p - 1))
    eigenvalues = np.linalg.eigvals(companion)
    max_modulus = np.max(np.abs(eigenvalues))
    print(f"Max eigenvalue modulus: {max_modulus:.4f}")
    if max_modulus >= 1.0:
        print("WARNING: VAR is not stable!")
    return eigenvalues
```

---

## Saving Results

Save all outputs so downstream notebooks can load them:

```python
import pickle

results = {
    'A_list': A_list,           # list of 7×7 arrays
    'c': c,                      # 7×1 array
    'Sigma_u': Sigma_u,          # 7×7 array
    'Psi': Psi,                  # list of H+1 7×7 arrays
    'residuals': residuals,      # (T-p) × 7 array
    'variable_names': ['gdp_yoy', 'ip_yoy', 'cpi_yoy', 'policy_rate',
                        'm2_yoy', 'neer_yoy', 'usip_yoy'],
    'dates': dates,              # array of dates corresponding to residuals
    'p': 6,                      # number of lags
    'H': 120,                    # Wold horizon
}

with open('bvar_results.pkl', 'wb') as f:
    pickle.dump(results, f)
```

---

## Summary: What Goes Where Downstream

| Object | Used in | Purpose |
|--------|---------|---------|
| Ψ_0, ..., Ψ_H (Wold coefficients) | Counterfactual (Step 3) | First sufficient statistic in Wolf et al. Proposition 1 |
| u_t (residuals) | Counterfactual (Step 3) | History of shocks to reweight under alternative policy |
| Forecasts from specific dates | Counterfactual (Step 3) | Baseline projection ("what would have happened without policy change") |
| A_1, ..., A_6 | Wold computation, forecasting | Intermediate — not used directly in counterfactual |
| Σ_u | Possibly in counterfactual | Needed if orthogonalizing residuals |
