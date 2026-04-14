"""
Stock & Watson (2010) Monthly GDP Distribution
================================================
Adapted to plug into gdp_monthly_df from Data_Montly.ipynb.

Usage:
    1. Run your notebook cells up to the point where gdp_monthly_df is built.
    2. Save: gdp_monthly_df.to_csv("gdp_monthly_df.csv", index=False)
    3. Paste these cells into the "Stock & Watson (2010)" section of your notebook.

Expected columns in gdp_monthly_df:
    - date              (monthly, datetime)
    - realgdp           (quarterly real GDP, non-NaN only on quarter-end months)
    - current consumption
    - proxy_fai
    - gov_spend_current
    - trade balance
    - CNYUSDSpot
    - cpi

Simple idea:
    We treat quarterly GDP as the sum of three monthly GDP values.
    Monthly indicators help split each quarterly total into three parts.

In short:
    Q_t = q_{t,1} + q_{t,2} + q_{t,3}

The code estimates a smooth monthly path that matches this identity.
"""

# ── Cell 1: Imports ──────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


# ── Cell 2: Core functions ───────────────────────────────────────────────────

def estimate_trend_quarterly(Q: np.ndarray):
    """
    Build a smooth quarterly trend and its monthly counterpart.

    The quarterly trend S_T is the sum of three monthly trend values:
        S_T = s_{3T-2} + s_{3T-1} + s_{3T}

    This lets us work with a detrended quarterly series:
        Q_tilde_T = Q_T / S_T
    """
    n_q = len(Q)
    t_q = np.arange(n_q, dtype=float)
    t_m = np.arange(n_q * 3) / 3.0

    log_Q = np.log(np.maximum(Q, 1e-10))
    cs = CubicSpline(t_q, log_Q)

    s_monthly = np.exp(cs(t_m))

    # S_quarterly = sum of 3 monthly trends within each quarter
    S_quarterly = np.array([
        s_monthly[3*i] + s_monthly[3*i+1] + s_monthly[3*i+2]
        for i in range(n_q)
    ])

    return S_quarterly, s_monthly


def kalman_smoother(Q_tilde, x_m, s_m, S_q, beta, rho, sigma_eps):
    """
    Kalman filter + RTS smoother for the monthly GDP path.

    Monthly model:
        q_t = mu_t + u_t
        u_t = rho * u_{t-1} + eps_t

    Quarterly adding-up:
        Q_T = q_{3T-2} + q_{3T-1} + q_{3T}

    The quarterly observation helps pin down the three monthly values.

    Returns (q_tilde_smoothed, log_likelihood).
    """
    n_m = len(x_m)
    if x_m.ndim == 1:
        x_m = x_m.reshape(-1, 1)

    # Deterministic component
    mu = beta[0] + x_m @ beta[1:]  # (n_m,)

    sigma2 = sigma_eps ** 2

    # --- Initialise filter arrays ---
    a_pred = np.zeros(n_m)
    P_pred = np.zeros(n_m)
    a_filt = np.zeros(n_m)
    P_filt = np.zeros(n_m)

    # Unconditional / diffuse init
    a_pred[0] = 0.0
    P_pred[0] = sigma2 / (1 - rho**2) if abs(rho) < 0.999 else sigma2 * 100

    log_lik = 0.0

    # --- Forward pass (filter) ---
    for t in range(n_m):
        if t > 0:
            a_pred[t] = rho * a_filt[t - 1]
            P_pred[t] = rho**2 * P_filt[t - 1] + sigma2

        month_in_q = t % 3

        if month_in_q < 2:
            # No observation yet — prediction = filtered
            a_filt[t] = a_pred[t]
            P_filt[t] = P_pred[t]
        else:
            # End of quarter: we observe Q_tilde
            T_idx = t // 3

            # Quarterly GDP is the sum of three monthly values.
            # We use trend-based weights to split the quarterly total.
            w = np.array([
                s_m[t - 2] / S_q[T_idx],
                s_m[t - 1] / S_q[T_idx],
                s_m[t]     / S_q[T_idx],
            ])

            # Build the 3-month block covariance
            a_blk = np.array([a_pred[t-2], a_pred[t-1], a_pred[t]])
            mu_blk = np.array([mu[t-2], mu[t-1], mu[t]])

            P_blk = np.zeros((3, 3))
            P_blk[0, 0] = P_pred[t - 2]
            P_blk[1, 1] = P_pred[t - 1]
            P_blk[2, 2] = P_pred[t]
            P_blk[0, 1] = P_blk[1, 0] = rho * P_pred[t - 2]
            P_blk[0, 2] = P_blk[2, 0] = rho**2 * P_pred[t - 2]
            P_blk[1, 2] = P_blk[2, 1] = rho * P_pred[t - 1]

            # Innovation
            y_pred = w @ (mu_blk + a_blk)
            v = Q_tilde[T_idx] - y_pred
            F = w @ P_blk @ w
            F = max(F, 1e-15)

            # Kalman update (3-month block)
            K = P_blk @ w / F
            a_upd = a_blk + K * v
            P_upd = P_blk - np.outer(K, K) * F

            a_filt[t - 2], a_filt[t - 1], a_filt[t] = a_upd
            P_filt[t - 2], P_filt[t - 1], P_filt[t] = P_upd[0,0], P_upd[1,1], P_upd[2,2]

            log_lik += -0.5 * (np.log(2*np.pi) + np.log(F) + v**2 / F)

    # --- Backward pass (RTS smoother) ---
    a_smooth = np.copy(a_filt)
    P_smooth = np.copy(P_filt)

    for t in range(n_m - 2, -1, -1):
        if P_pred[t + 1] > 1e-15:
            J = rho * P_filt[t] / P_pred[t + 1]
        else:
            J = 0.0
        a_smooth[t] = a_filt[t] + J * (a_smooth[t + 1] - a_pred[t + 1])
        P_smooth[t] = P_filt[t] + J**2 * (P_smooth[t + 1] - P_pred[t + 1])

    return mu + a_smooth, log_lik


def neg_loglik(params, Q_tilde, x_m, s_m, S_q, k):
    """Negative log-likelihood for scipy.optimize.minimize."""
    beta = params[:k + 1]
    rho = np.tanh(params[k + 1])         # (-1, 1)
    sigma_eps = np.exp(params[k + 2])    # > 0
    try:
        _, ll = kalman_smoother(Q_tilde, x_m, s_m, S_q, beta, rho, sigma_eps)
        return -ll
    except Exception:
        return 1e10


def distribute_quarterly(Q_quarterly, x_monthly):
    """
    Split quarterly GDP into monthly estimates using monthly indicators.

    Parameters
    ----------
    Q_quarterly : 1-d array, shape (n_q,)
    x_monthly   : 2-d array, shape (n_q * 3, k)

    Returns
    -------
    q_monthly : 1-d array, shape (n_q * 3,)

    Basic idea:
        1. build a smooth quarterly trend,
        2. use monthly indicators to guide the split,
        3. make the three monthly values add up to quarterly GDP.
    """
    n_q = len(Q_quarterly)
    n_m = n_q * 3
    assert len(x_monthly) == n_m

    if x_monthly.ndim == 1:
        x_monthly = x_monthly.reshape(-1, 1)
    k = x_monthly.shape[1]

    # 1. Smooth trends
    S_q, s_m = estimate_trend_quarterly(Q_quarterly)
    Q_tilde = Q_quarterly / S_q

    # 2. Detrend the monthly indicators with their own smooth trends
    x_det = np.zeros_like(x_monthly)
    for j in range(k):
        col = x_monthly[:, j]
        # Quarterly average of the indicator
        xq = np.array([col[3*i : 3*i+3].mean() for i in range(n_q)])
        _, sx = estimate_trend_quarterly(np.maximum(xq, 1e-10))
        x_det[:, j] = col / sx

    # 3. Estimate the model parameters
    beta0 = np.zeros(k + 1)
    beta0[0] = np.mean(Q_tilde)
    params0 = np.concatenate([beta0, [np.arctanh(0.5), np.log(np.std(Q_tilde) * 0.1 + 1e-6)]])

    res = minimize(
        neg_loglik, params0,
        args=(Q_tilde, x_det, s_m, S_q, k),
        method="Nelder-Mead",
        options={"maxiter": 15000, "xatol": 1e-8, "fatol": 1e-8},
    )
    beta_hat = res.x[:k + 1]
    rho_hat  = np.tanh(res.x[k + 1])
    sig_hat  = np.exp(res.x[k + 2])
    print(f"  β = {np.round(beta_hat, 4)},  ρ = {rho_hat:.4f},  σ = {sig_hat:.6f}")

    # 4. Recover the monthly path
    q_tilde_sm, _ = kalman_smoother(Q_tilde, x_det, s_m, S_q, beta_hat, rho_hat, sig_hat)

    # 5. Put the trend back
    q_monthly = q_tilde_sm * s_m

    # 6. Force exact quarterly adding-up
    for i in range(n_q):
        m0 = 3 * i
        total = q_monthly[m0 : m0 + 3].sum()
        if abs(total) > 1e-10:
            q_monthly[m0 : m0 + 3] *= Q_quarterly[i] / total

    return q_monthly


# ── Cell 3: Prepare data from gdp_monthly_df ────────────────────────────────

def _extract_subsample(df, indicator_cols, start_date, end_date):
    """
    Extract a clean subsample: trim to [start_date, end_date],
    ensure n_months is a multiple of 3, interpolate indicators.
    Returns (sub_df, Q_array, x_array, n_q, n_m).
    """
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    sub = df.loc[mask].copy().reset_index(drop=True)

    # Drop rows at the end if indicators are missing
    for c in indicator_cols:
        last_valid = sub[c].last_valid_index()
        if last_valid is not None:
            sub = sub.iloc[: last_valid + 1]

    # Ensure n_months is a multiple of 3
    n_m = len(sub)
    n_m = n_m - (n_m % 3)
    sub = sub.iloc[:n_m].reset_index(drop=True)
    n_q = n_m // 3

    # Quarterly GDP
    Q = sub.loc[sub["realgdp"].notna(), "realgdp"].values
    assert len(Q) == n_q, f"Expected {n_q} quarterly obs, got {len(Q)}"

    # Monthly indicators
    x = sub[indicator_cols].interpolate(method="linear", limit_direction="both").values
    # Fill any remaining NaN with column mean
    for j in range(x.shape[1]):
        nans = np.isnan(x[:, j])
        if nans.any():
            x[nans, j] = np.nanmean(x[:, j])

    return sub, Q, x, n_q, n_m


def _seasonal_adjust_monthly(series: np.ndarray, dates: pd.Series) -> np.ndarray:
    """
    Simple multiplicative seasonal adjustment for a monthly series.
    Estimates seasonal factors as the average ratio of each month
    to a centered 12-month moving average, then divides them out.

    Handles negative values (e.g., trade balance) by using additive
    adjustment instead.
    """
    s = pd.Series(series, index=dates).copy()

    if (s <= 0).any():
        # Additive seasonal adjustment for series with negatives
        ma12 = s.rolling(12, center=True, min_periods=6).mean()
        diff = s - ma12
        seasonal = diff.groupby(diff.index.month).transform("mean")
        sa = s - seasonal
    else:
        # Multiplicative seasonal adjustment
        ma12 = s.rolling(12, center=True, min_periods=6).mean()
        ratio = s / ma12
        seasonal = ratio.groupby(ratio.index.month).transform("mean")
        sa = s / seasonal

    return sa.values


def _seasonal_adjust_quarterly(Q: np.ndarray, n_q: int) -> np.ndarray:
    """
    Remove quarterly seasonal pattern (Q4 > Q1 etc.) from quarterly series.
    Uses ratio-to-moving-average with a 4-quarter centered MA.
    """
    s = pd.Series(Q)

    if n_q < 8:
        return Q  # not enough data to estimate seasonal factors

    ma4 = s.rolling(4, center=True, min_periods=2).mean()
    ratio = s / ma4
    # Quarter index: 0, 1, 2, 3 repeating
    q_idx = np.arange(n_q) % 4
    seasonal_factors = np.array([ratio[q_idx == q].mean() for q in range(4)])

    # Normalize so factors average to 1
    seasonal_factors /= seasonal_factors.mean()

    return Q / seasonal_factors[q_idx]


def prepare_and_run(df: pd.DataFrame, break_year: int = None) -> pd.DataFrame:
    """
    Takes your gdp_monthly_df, aligns quarterly GDP with monthly indicators,
    and runs the Stock-Watson distribution.

    IMPORTANT: Stock-Watson assumes all input series are seasonally adjusted.
    This function applies seasonal adjustment to both the quarterly GDP
    and the monthly indicators before distribution, then re-applies the
    seasonal pattern to the final monthly estimates.

    If break_year is set, the sample is split into two sub-periods,
    each estimated with its own trend and parameters.

    Set break_year=None to run the full sample as one block.

    Returns a copy of df with new columns 'realgdp_monthly' and
    'realgdp_monthly_yoy'.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    indicator_cols = ["current consumption", "proxy_fai", "gov_spend_current", "trade balance"]

    # Convert trade balance from USD to RMB if spot rate available
    if "CNYUSDSpot" in df.columns:
        mask = df["trade balance"].notna() & df["CNYUSDSpot"].notna()
        df.loc[mask, "trade balance"] = (
            df.loc[mask, "trade balance"].astype(float)
            * df.loc[mask, "CNYUSDSpot"].astype(float)
        )

    # Ensure numeric
    for c in indicator_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["realgdp"] = pd.to_numeric(df["realgdp"], errors="coerce")

    # Quarter-end months with GDP
    q_mask = df["realgdp"].notna()
    q_dates = df.loc[q_mask, "date"]
    first_q = q_dates.iloc[0]
    last_q  = q_dates.iloc[-1]
    first_m = first_q - pd.DateOffset(months=2)

    # ── Define sub-periods ───────────────────────────────────────────────
    if break_year is not None:
        # Period 1: start to Dec of break_year
        # Period 2: Jan of break_year+1 to end
        # We need each period to start on the 1st month of a quarter
        # Period 1 ends at the last quarter-end of break_year
        break_end   = pd.Timestamp(f"{break_year}-12-01")
        break_start = pd.Timestamp(f"{break_year + 1}-01-01")

        periods = [
            (f"Period 1 (–{break_year})", first_m, break_end),
            (f"Period 2 ({break_year+1}–)", break_start, last_q),
        ]
    else:
        periods = [
            ("Full sample", first_m, last_q),
        ]

    # ── Run each sub-period ──────────────────────────────────────────────
    all_results = []

    for label, start, end in periods:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        sub, Q, x, n_q, n_m = _extract_subsample(df, indicator_cols, start, end)

        print(f"Sample: {sub['date'].iloc[0].strftime('%Y-%m')} to "
              f"{sub['date'].iloc[-1].strftime('%Y-%m')}  "
              f"({n_q} quarters, {n_m} months)")

        # Check for NaN warnings
        nan_counts = np.isnan(x).sum(axis=0)
        for i, c in enumerate(indicator_cols):
            if nan_counts[i] > 0:
                print(f"  WARNING: {c} still has {nan_counts[i]} NaN after interpolation")

        # ── Seasonal adjustment ──────────────────────────────────────────
        # Stock-Watson assumes all series are seasonally adjusted.
        # 1. SA the quarterly GDP
        Q_sa = _seasonal_adjust_quarterly(Q, n_q)
        # Save quarterly seasonal factors for re-seasonalising later
        q_idx = np.arange(n_q) % 4
        q_seasonal = np.array([
            (Q / Q_sa)[q_idx == q].mean() for q in range(4)
        ])
        print(f"  Quarterly seasonal factors: {np.round(q_seasonal, 3)}")

        # 2. SA the monthly indicators
        x_sa = np.zeros_like(x)
        dates = sub["date"]
        for j in range(x.shape[1]):
            x_sa[:, j] = _seasonal_adjust_monthly(x[:, j], dates)

        # ── Distribute (on SA data) ──────────────────────────────────────
        print("\nEstimating monthly real GDP (seasonally adjusted)...")
        q_monthly_sa = distribute_quarterly(Q_sa, x_sa)

        # ── Re-apply seasonal pattern to monthly estimates ───────────────
        # Distribute the quarterly seasonal factor evenly across the 3 months
        month_seasonal = np.ones(n_m)
        for i in range(n_q):
            qi = i % 4
            month_seasonal[3*i : 3*i+3] = q_seasonal[qi]
        q_monthly = q_monthly_sa * month_seasonal

        # Re-enforce adding-up to original (not SA) quarterly GDP
        for i in range(n_q):
            m0 = 3 * i
            total = q_monthly[m0 : m0 + 3].sum()
            if abs(total) > 1e-10:
                q_monthly[m0 : m0 + 3] *= Q[i] / total

        # Verify adding-up
        print("\nAdding-up check (first 5 quarters):")
        for i in range(min(5, n_q)):
            m0 = 3 * i
            total = q_monthly[m0 : m0 + 3].sum()
            print(f"  Q{i+1}: quarterly = {Q[i]:.2f}, monthly sum = {total:.2f}, "
                  f"diff = {Q[i] - total:.8f}")

        sub["realgdp_monthly"] = q_monthly
        all_results.append(sub[["date", "realgdp_monthly"]])

    # ── Combine results and merge back ───────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)

    # Drop old results if re-running
    for col in ["realgdp_monthly", "realgdp_monthly_yoy"]:
        if col in df.columns:
            df = df.drop(columns=col)

    df = df.merge(combined, on="date", how="left")

    # ── YoY growth ───────────────────────────────────────────────────────
    df = df.sort_values("date").reset_index(drop=True)
    df["realgdp_monthly_yoy"] = df["realgdp_monthly"].pct_change(12) * 100

    return df


# ── Cell 4: Run it ───────────────────────────────────────────────────────────

# In your notebook, just do:
#
#   gdp_monthly_df = prepare_and_run(gdp_monthly_df)
#
#   # Inspect results
#   print(gdp_monthly_df[["date", "realgdp_monthly", "realgdp_monthly_yoy"]].dropna().head(24))
#
#   # Quick plot
#   import matplotlib.pyplot as plt
#   fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#
#   ax = axes[0]
#   ax.plot(gdp_monthly_df["date"], gdp_monthly_df["realgdp_monthly"], label="Monthly (distributed)")
#   q_rows = gdp_monthly_df["realgdp"].notna()
#   ax.scatter(gdp_monthly_df.loc[q_rows, "date"], gdp_monthly_df.loc[q_rows, "realgdp"],
#              color="red", zorder=5, label="Quarterly (NBS)")
#   ax.set_ylabel("Real GDP (亿元)")
#   ax.legend()
#   ax.set_title("Monthly Real GDP — Stock & Watson (2010) Distribution")
#
#   ax = axes[1]
#   ax.plot(gdp_monthly_df["date"], gdp_monthly_df["realgdp_monthly_yoy"])
#   ax.axhline(0, color="grey", linewidth=0.5)
#   ax.set_ylabel("YoY growth (%)")
#   ax.set_title("Monthly Real GDP Growth (YoY)")
#
#   plt.tight_layout()
#   plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST (run this file directly to verify with synthetic data)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    n_q = 40
    n_m = n_q * 3

    trend = np.exp(np.linspace(np.log(50000), np.log(120000), n_q))
    Q = trend + 2000 * np.sin(np.linspace(0, 8*np.pi, n_q)) + np.random.randn(n_q) * 500

    t = np.arange(n_m)
    x1 = np.exp(np.linspace(np.log(30000), np.log(75000), n_m)) * (1 + 0.02*np.sin(2*np.pi*t/12))
    x2 = np.exp(np.linspace(np.log(15000), np.log(40000), n_m)) * (1 + 0.03*np.sin(2*np.pi*t/12))
    x3 = np.exp(np.linspace(np.log(8000),  np.log(20000), n_m)) * (1 + 0.04*np.sin(2*np.pi*t/12))
    x4 = 2000 + 500*np.sin(2*np.pi*t/12) + np.random.randn(n_m) * 100

    x = np.column_stack([x1, x2, x3, x4])

    print("Synthetic data test:")
    q_monthly = distribute_quarterly(Q, x)

    print("\nAdding-up check:")
    for i in range(5):
        m0 = 3 * i
        total = q_monthly[m0:m0+3].sum()
        print(f"  Q{i+1}: {Q[i]:.2f} vs {total:.2f}, diff = {Q[i]-total:.8f}")
