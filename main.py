# filename: sharpe_lstm_rolling_buffer_prices_returns_tc_gross_with_benchmarks_TLT_allocplots.py
import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers  # type: ignore

# Try to import scipy; provide a simple fallback if not available
try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# =========================
# Reproducibility
# =========================
SEED = 420
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Parameters
# =========================
tickers = ['VTI', 'AGG', 'DBC', 'TLT']
ticker_rename = {'VTI': 'S', 'AGG': 'B', 'DBC': 'C', 'TLT': 'L'}

start_date = '2006-01-01'
end_date   = '2020-04-30'

# Model / training
window        = 50
lstm_units    = 64
epochs        = 100
learning_rate = 1e-3
epsilon       = 1e-8
retrain_interval_years = 1   # change to 2 if you want retraining every 2 years

# Transaction cost
cost_rate = 0.0001  # 0.01% per unit of abs turnover

# Rolling config
first_test_year = 2011
last_test_year  = 2020
buffer_days     = 50

# Static Benchmarks
BM_WEIGHTS = {
    "BM1_25_25_25_25": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
    "BM2_50_10_20_20": np.array([0.50, 0.10, 0.20, 0.20], dtype=np.float32),
    "BM3_10_50_20_20": np.array([0.10, 0.50, 0.20, 0.20], dtype=np.float32),
    "BM4_40_40_10_10": np.array([0.40, 0.40, 0.10, 0.10], dtype=np.float32),
}

# Output
save_dir = './results_rolling_buffer'
os.makedirs(save_dir, exist_ok=True)

assert window <= buffer_days, (
    f"window ({window}) must be <= buffer_days ({buffer_days}) "
)

# =========================
# 1) Data download & preprocessing
# =========================
print("Downloading prices...")
prices = yf.download(tickers, start=start_date, end=end_date)['Close']
prices = prices.rename(columns=ticker_rename).dropna(how='all').sort_index()

desired_order = ['S', 'B', 'C', 'L']
available = [c for c in desired_order if c in prices.columns]
prices = prices[available].dropna()

rets = prices.pct_change().dropna()
prices_aligned = prices.loc[rets.index, desired_order]

assets   = desired_order
n_assets = len(assets)
print(f"Assets (ordered): {assets}, rows: {len(prices_aligned)}")

features_df = pd.concat(
    [prices_aligned.add_prefix('price_'), rets[assets].add_prefix('ret_')],
    axis=1
)
targets_df = rets[assets].copy()
n_features = features_df.shape[1]

# =========================
# Helpers
# =========================
def build_windows(features: pd.DataFrame, targets: pd.DataFrame, lookback: int):
    f = features.values
    t = targets.values
    X, Y, idxs = [], [], []
    for i in range(lookback, len(f)):
        X.append(f[i-lookback:i])
        Y.append(t[i])
        idxs.append(features.index[i])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), pd.DatetimeIndex(idxs)

def build_model(input_shape, lstm_units: int, outputs: int):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(lstm_units)(inp)
    x = layers.Dense(outputs)(x)
    out = layers.Softmax()(x)
    return models.Model(inputs=inp, outputs=out)

@tf.function
def sharpe_loss_gross(y_true, y_pred):
    port_rets = tf.reduce_sum(y_true * y_pred, axis=1)
    mean = tf.reduce_mean(port_rets)
    std  = tf.math.reduce_std(port_rets)
    sharpe = mean / (std + epsilon)
    loss = -sharpe
    return loss, mean, std, sharpe

def summary_stats(simple_rets: np.ndarray):
    ann = 252.0
    mu  = np.nanmean(simple_rets)
    sd  = np.nanstd(simple_rets)
    sharpe_ann = (mu / (sd + epsilon)) * np.sqrt(ann)
    cum_ret = np.prod(1.0 + simple_rets) - 1.0
    return {'mean_daily': mu, 'std_daily': sd, 'sharpe_ann': sharpe_ann, 'cum_ret': cum_ret}

# --- Metrics helpers (ANNUALIZED) ---
def downside_deviation_daily(returns: np.ndarray, mar: float = 0.0) -> float:
    """Daily downside deviation relative to MAR (default 0)."""
    r = np.asarray(returns, dtype=float)
    downside = np.minimum(r - mar, 0.0)
    return float(np.sqrt(np.mean(downside**2) + 1e-18))

def max_drawdown_from_nav(nav: np.ndarray) -> float:
    """Maximum drawdown magnitude from a NAV path (e.g., 0.34 = -34%)."""
    nav = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.clip(peak, 1e-18, None) - 1.0
    return float(-np.min(dd))

def compute_metrics_annualized(returns: np.ndarray, nav: np.ndarray, ann_factor: float = 252.0) -> dict:
    """
    Compute ANNUALIZED metrics:
      E(R), Std(R), Sharpe, DD(R), Sortino, and MDD (from NAV).
    All except MDD are annualized from daily returns.
    """
    r = np.asarray(returns, dtype=float)
    mu_daily = float(np.nanmean(r))
    sd_daily = float(np.nanstd(r) + 1e-18)
    dd_daily = downside_deviation_daily(r)

    mu_ann   = mu_daily * ann_factor
    sd_ann   = sd_daily * np.sqrt(ann_factor)
    sharpe   = (mu_daily / sd_daily) * np.sqrt(ann_factor)
    dd_ann   = dd_daily * np.sqrt(ann_factor)
    sortino  = (mu_daily / (dd_daily + 1e-18)) * np.sqrt(ann_factor)
    mdd      = max_drawdown_from_nav(nav)

    return {
        "E(R)": mu_ann,       # annualized mean return
        "Std(R)": sd_ann,     # annualized volatility
        "Sharpe": sharpe,     # annualized Sharpe
        "DD(R)": dd_ann,      # annualized downside deviation
        "Sortino": sortino,   # annualized Sortino
        "MDD": mdd            # max drawdown (fraction)
    }

# ---------- Optimizers for MV and MD ----------
def _solve_qp_ratio(objective, w0, bounds, cons):
    if HAVE_SCIPY:
        res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
        w = res.x
        w[w < 0] = 0
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w0) / len(w0)
        else:
            w = w / s
        return w
    else:
        return None

def solve_mv(mu, Sigma, w_prev=None):
    n = len(mu)
    eps = 1e-10
    def obj(w):
        num = float(np.dot(w, mu))
        den = float(np.sqrt(max(np.dot(w, Sigma @ w), eps)))
        return - num / den
    w0 = np.ones(n) / n if w_prev is None else w_prev
    bounds = [(0.0, 1.0)] * n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    w = _solve_qp_ratio(obj, w0, bounds, cons)
    if w is None:
        try:
            sig = np.linalg.pinv(Sigma) @ mu
        except Exception:
            sig = np.ones(n)
        sig[sig < 0] = 0
        s = sig.sum()
        w = (sig / s) if s > 0 else np.ones(n)/n
    return w

def solve_md(sigma_vec, Sigma, w_prev=None):
    n = len(sigma_vec)
    eps = 1e-10
    def obj(w):
        num = float(np.dot(w, sigma_vec))
        den = float(np.sqrt(max(np.dot(w, Sigma @ w), eps)))
        return - num / den
    w0 = np.ones(n) / n if w_prev is None else w_prev
    bounds = [(0.0, 1.0)] * n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    w = _solve_qp_ratio(obj, w0, bounds, cons)
    if w is None:
        try:
            sig = np.linalg.pinv(Sigma) @ sigma_vec
        except Exception:
            sig = np.ones(n)
        sig[sig < 0] = 0
        s = sig.sum()
        w = (sig / s) if s > 0 else np.ones(n)/n
    return w

# =========================
# 2) Rolling walk-forward + Benchmarks
# =========================
all_allocations = []
all_performance = []
mv_allocations = []
md_allocations = []

nav = 1.0
prev_weights_carry = np.ones(n_assets, dtype=np.float32) / n_assets
model = None

bm_prev_w = {name: np.ones(n_assets, dtype=np.float32) / n_assets for name in BM_WEIGHTS}
bm_nav    = {name: 1.0 for name in BM_WEIGHTS}

mv_prev_w = np.ones(n_assets, dtype=np.float32) / n_assets
md_prev_w = np.ones(n_assets, dtype=np.float32) / n_assets
mv_nav = 1.0
md_nav = 1.0

for test_year in range(first_test_year, last_test_year + 1):
    prev_year_end = f"{test_year-1}-12-31"
    test_start    = f"{test_year}-01-01"
    test_end      = f"{test_year}-12-31"

    feats_until_prev   = features_df.loc[:prev_year_end]
    targets_until_prev = targets_df.loc[:prev_year_end]

    if len(feats_until_prev) <= buffer_days + window:
        print(f"Skipping {test_year}: not enough history to respect buffer + window.")
        continue

    # --- Train set: EXCLUDE last 50 trading days of (Y-1)
    train_feats   = feats_until_prev.iloc[: -buffer_days]
    train_targets = targets_until_prev.iloc[: -buffer_days]

    # --- Lookback + Test: last 50 of (Y-1) + all of Y
    lookback_start = feats_until_prev.index[-buffer_days]
    feats_lookback_and_test   = features_df.loc[lookback_start : test_end]
    targets_lookback_and_test = targets_df.loc[lookback_start : test_end]

    # Build windows
    X_train, Y_train, _ = build_windows(train_feats, train_targets, window)
    if len(X_train) == 0:
        print(f"Skipping {test_year}: no train windows after buffer.")
        continue

    X_test_all, Y_test_all, idx_all = build_windows(feats_lookback_and_test, targets_lookback_and_test, window)
    mask_year = (idx_all >= pd.to_datetime(test_start)) & (idx_all <= pd.to_datetime(test_end))
    X_test, Y_test, test_idx = X_test_all[mask_year], Y_test_all[mask_year], idx_all[mask_year]
    if len(X_test) == 0:
        print(f"Skipping {test_year}: no test windows inside the year.")
        continue

    print(f"\n=== Train <= {prev_year_end} (EXCLUDING last {buffer_days} days), Test {test_year} ===")
    print(f" Train windows: {len(X_train)} | Test windows: {len(X_test)}")

    # ----- Train DL model only every N years -----
    if ((test_year - first_test_year) % retrain_interval_years) == 0:
        print(f" Retraining DL model for {test_year}...")
        model = build_model((window, n_features), lstm_units, n_assets)
        opt = optimizers.Adam(learning_rate=learning_rate)

        X_train_tf = tf.constant(X_train)
        Y_train_tf = tf.constant(Y_train)

        for epoch in range(1, epochs + 1):
            with tf.GradientTape() as tape:
                preds = model(X_train_tf, training=True)
                loss, mean_ret, std_ret, sharpe_val = sharpe_loss_gross(Y_train_tf, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            if epoch % 10 == 0 or epoch == 1:
                print(f" Epoch {epoch:03d} | loss={float(loss):.6f} "
                      f"| sharpe(gross)={float(sharpe_val):.6f} "
                      f"| mean={float(mean_ret):.3e} | std={float(std_ret):.3e}")
    else:
        print(f" Reusing previous DL model for {test_year}...")

    # ----- Predict allocations for the test year (DL strategy) -----
    preds_test = model.predict(X_test, verbose=0)
    alloc_df = pd.DataFrame(preds_test, index=test_idx, columns=assets)
    all_allocations.append(alloc_df)

    # ----- Evaluate DL with transaction costs -----
    weights = preds_test
    w_prev = prev_weights_carry
    deltas = np.vstack([weights[0] - w_prev, weights[1:] - weights[:-1]])   # (n_days, n_assets)
    tc_dl = cost_rate * np.sum(np.abs(deltas), axis=1)                      # (n_days,)

    r_gross_dl = np.sum(weights * Y_test, axis=1)
    r_net_dl   = r_gross_dl - tc_dl

    prev_weights_carry = weights[-1].astype(np.float32)

    port_rets_s = pd.Series(r_net_dl, index=test_idx, name='DL_Portfolio')
    nav_series_dl = (1.0 + port_rets_s).cumprod() * nav
    nav = float(nav_series_dl.iloc[-1])

    # ----- Static Benchmarks (yearly rebalancing) -----
    bm_nav_series = {}
    bm_net_rets   = {}
    for name, w_target in BM_WEIGHTS.items():
        n_days = len(test_idx)
        r_gross_bm = Y_test @ w_target
        delta0 = np.sum(np.abs(w_target - bm_prev_w[name]))
        tc_bm = np.zeros(n_days, dtype=np.float32)
        tc_bm[0] = cost_rate * delta0
        r_net_bm = r_gross_bm - tc_bm
        bm_net_rets[name] = r_net_bm
        nav_series = (1.0 + r_net_bm).cumprod() * bm_nav[name]
        bm_nav_series[name] = pd.Series(nav_series, index=test_idx)
        bm_prev_w[name] = w_target.astype(np.float32)
        bm_nav[name]    = float(nav_series[-1])

    # ----- Dynamic daily MV & MD (50-day rolling μ/Σ) -----
    rets_ref = targets_lookback_and_test  # includes the 50-day pre-year buffer
    mv_net = np.zeros(len(test_idx), dtype=np.float64)
    md_net = np.zeros(len(test_idx), dtype=np.float64)
    mv_nav_series = np.zeros(len(test_idx), dtype=np.float64)
    md_nav_series = np.zeros(len(test_idx), dtype=np.float64)
    w_mv_prev = mv_prev_w.copy()
    w_md_prev = md_prev_w.copy()
    cum_mv = mv_nav
    cum_md = md_nav

    # keep weight paths for this year
    mv_w_year = []
    md_w_year = []

    for i, d in enumerate(test_idx):
        # last 50 days BEFORE day d
        hist = rets_ref.loc[:d].iloc[:-1].tail(50)
        if len(hist) < 2:
            mu = np.zeros(n_assets)
            Sigma = np.eye(n_assets) * 1e-6
        else:
            mu = hist.mean().values
            Sigma = hist.cov().values
            Sigma = Sigma + np.eye(n_assets) * 1e-8  # regularize

        sigma_vec = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))

        w_mv = solve_mv(mu, Sigma, w_prev=w_mv_prev)
        w_md = solve_md(sigma_vec, Sigma, w_prev=w_md_prev)

        # store today's weights
        mv_w_year.append(w_mv.copy())
        md_w_year.append(w_md.copy())

        rt = Y_test[i]
        r_gross_mv = float(np.dot(w_mv, rt))
        r_gross_md = float(np.dot(w_md, rt))

        tc_mv = cost_rate * float(np.sum(np.abs(w_mv - w_mv_prev)))
        tc_md = cost_rate * float(np.sum(np.abs(w_md - w_md_prev)))

        r_net_mv = r_gross_mv - tc_mv
        r_net_md = r_gross_md - tc_md

        mv_net[i] = r_net_mv
        md_net[i] = r_net_md

        cum_mv *= (1.0 + r_net_mv)
        cum_md *= (1.0 + r_net_md)
        mv_nav_series[i] = cum_mv
        md_nav_series[i] = cum_md

        w_mv_prev = w_mv.astype(np.float32)
        w_md_prev = w_md.astype(np.float32)

    # carry across years
    mv_prev_w = w_mv_prev
    md_prev_w = w_md_prev
    mv_nav = float(mv_nav_series[-1])
    md_nav = float(md_nav_series[-1])

    # save this year's MV/MD weight paths
    mv_w_df = pd.DataFrame(np.vstack(mv_w_year), index=test_idx, columns=assets)
    md_w_df = pd.DataFrame(np.vstack(md_w_year), index=test_idx, columns=assets)
    mv_allocations.append(mv_w_df)
    md_allocations.append(md_w_df)

    # ----- Assemble yearly performance frame -----
    perf_df = pd.DataFrame({
        'GrossRet_DL': r_gross_dl,
        'TurnoverCost_DL': tc_dl,
        'DL_NetRet': port_rets_s.values,
        'DL_NAV': nav_series_dl.values,
        # Dynamic benchmarks:
        'MV_NetRet': mv_net,
        'MV_NAV'   : mv_nav_series,
        'MD_NetRet': md_net,
        'MD_NAV'   : md_nav_series,
    }, index=test_idx)

    # Add static benchmark NAVs/returns
    for name in BM_WEIGHTS:
        perf_df[f'{name}_NetRet'] = bm_net_rets[name]
        perf_df[f'{name}_NAV']    = bm_nav_series[name].values

    all_performance.append(perf_df)

# =========================
# 3) Concatenate & save performance + allocations
# =========================
if len(all_allocations) == 0:
    raise RuntimeError("No allocations generated. Check data availability or parameters.")

alloc_dl = pd.concat(all_allocations).sort_index()           # DL weights across all years
alloc_mv = pd.concat(mv_allocations).sort_index()
alloc_md = pd.concat(md_allocations).sort_index()

perf_all  = pd.concat(all_performance).sort_index()

# Save
alloc_path = os.path.join(save_dir, 'allocations_DL.csv')
alloc_mv_path = os.path.join(save_dir, 'allocations_MV.csv')
alloc_md_path = os.path.join(save_dir, 'allocations_MD.csv')
perf_path = os.path.join(save_dir, 'performance_2011_2020_TLT.csv')

alloc_dl.to_csv(alloc_path)
alloc_mv.to_csv(alloc_mv_path)
alloc_md.to_csv(alloc_md_path)
perf_all.to_csv(perf_path)

print(f"\nSaved:\n - {alloc_path}\n - {alloc_mv_path}\n - {alloc_md_path}\n - {perf_path}")

# =========================
# 3b) Metrics table for all strategies (ANNUALIZED)
# =========================
metrics_rows = []

core = [
    ("DL", "DL_NetRet", "DL_NAV"),
    ("MV", "MV_NetRet", "MV_NAV"),
    ("MD", "MD_NetRet", "MD_NAV"),
]
for label, ret_col, nav_col in core:
    m = compute_metrics_annualized(perf_all[ret_col].values, perf_all[nav_col].values)
    metrics_rows.append({"Strategy": label, **m})

for name in BM_WEIGHTS.keys():
    ret_col = f"{name}_NetRet"
    nav_col = f"{name}_NAV"
    m = compute_metrics_annualized(perf_all[ret_col].values, perf_all[nav_col].values)
    metrics_rows.append({"Strategy": name, **m})

# Order columns as requested
col_order = ["E(R)", "Std(R)", "Sharpe", "DD(R)", "Sortino", "MDD"]
metrics_df = pd.DataFrame(metrics_rows).set_index("Strategy")[col_order]

# Save CSV (annualized metrics)
metrics_path = os.path.join(save_dir, "metrics_2011_2020_annualized.csv")
metrics_df.to_csv(metrics_path)
print("\nSaved ANNUALIZED metrics table:", metrics_path)
print(metrics_df.round(6))

# Save LaTeX table (add a caption note about annualization when you \input it)
latex_path = os.path.join(save_dir, "metrics_2011_2020_annualized.tex")
with open(latex_path, "w") as f:
    f.write(metrics_df.round(4).to_latex())
print("Saved LaTeX table:", latex_path)


# =========================
# 4) Plot continuous NAVs (net of costs) — DL + Benchmarks (static + MV/MD)
# =========================
nav_cols = ['DL_NAV', 'MV_NAV', 'MD_NAV'] + [f'{name}_NAV' for name in BM_WEIGHTS]
nav_df = perf_all[nav_cols].copy()

# (A) Combined NAVs
plt.figure(figsize=(12, 6))
for col in nav_cols:
    label = (
        'Deep Learning (net)' if col == 'DL_NAV' else
        'MV (net, daily)'    if col == 'MV_NAV' else
        'MD (net, daily)'    if col == 'MD_NAV' else
        col.replace('_NAV', '').replace('_', ' ')
    )
    lw = 2 if col in ['DL_NAV','MV_NAV','MD_NAV'] else 1.3
    plt.plot(nav_df.index, nav_df[col], label=label, linewidth=lw)
plt.title('Continuous NAV (net of 0.01% TC)\nDL vs Static Benchmarks (yearly) vs MV/MD (daily) — S,B,C,L')
plt.xlabel('Date'); plt.ylabel('Cumulative Value (start=1.0)')
plt.grid(True); plt.legend()
plt.tight_layout()
plot_path_all = os.path.join(save_dir, 'nav_with_benchmarks_mv_md_TLT.png')
plt.savefig(plot_path_all, dpi=200)
plt.close()
print("Saved NAV comparison plot:", plot_path_all)

# (B) Faceted NAVs
n_series = len(nav_cols)
n_cols = 2
n_rows = int(np.ceil(n_series / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
axes = axes.flatten()
for i, col in enumerate(nav_cols):
    ax = axes[i]
    label = (
        'Deep Learning (net)' if col == 'DL_NAV' else
        'MV (net, daily)'    if col == 'MV_NAV' else
        'MD (net, daily)'    if col == 'MD_NAV' else
        col.replace('_NAV', '').replace('_', ' ')
    )
    ax.plot(nav_df.index, nav_df[col])
    ax.set_title(label)
    ax.grid(True)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle('Continuous NAV by Strategy (net of 0.01% TC) — S,B,C,L', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plot_path_facets = os.path.join(save_dir, 'nav_with_benchmarks_facets_mv_md_TLT.png')
plt.savefig(plot_path_facets, dpi=200)
plt.close()
print("Saved faceted NAV plot:", plot_path_facets)

# =========================
# 5) Quick summary (net of TC)
# =========================
def summarize_series(net_rets: np.ndarray):
    ann = 252.0
    mu  = np.nanmean(net_rets)
    sd  = np.nanstd(net_rets)
    sharpe = (mu / (sd + epsilon)) * np.sqrt(ann)
    cum = np.prod(1.0 + net_rets) - 1.0
    return cum, sharpe

dl_cum, dl_sh = summarize_series(perf_all['DL_NetRet'].values)
mv_cum, mv_sh = summarize_series(perf_all['MV_NetRet'].values)
md_cum, md_sh = summarize_series(perf_all['MD_NetRet'].values)

print("\nOverall 2011–2020 (net of TC):")
print(f"  DL  : CumRet={dl_cum:.4f}, AnnSharpe={dl_sh:.4f}")
print(f"  MV  : CumRet={mv_cum:.4f}, AnnSharpe={mv_sh:.4f}")
print(f"  MD  : CumRet={md_cum:.4f}, AnnSharpe={md_sh:.4f}")
for name in BM_WEIGHTS:
    cum, sh = summarize_series(perf_all[f'{name}_NetRet'].values)
    print(f"  {name}: CumRet={cum:.4f}, AnnSharpe={sh:.4f}")

# =========================
# 6) Cumulative returns (non-compounded sum, net of TC)
# =========================
ret_cols = ['DL_NetRet', 'MV_NetRet', 'MD_NetRet'] + [f'{name}_NetRet' for name in BM_WEIGHTS]
cumret_add = perf_all[ret_cols].cumsum()

plt.figure(figsize=(12, 6))
for col in ret_cols:
    label = (
        'Deep Learning (net)' if col == 'DL_NetRet' else
        'MV (net, daily)'    if col == 'MV_NetRet' else
        'MD (net, daily)'    if col == 'MD_NetRet' else
        col.replace('_NetRet', '').replace('_', ' ')
    )
    lw = 2 if col in ['DL_NetRet','MV_NetRet','MD_NetRet'] else 1.3
    plt.plot(cumret_add.index, cumret_add[col], label=label, linewidth=lw)
plt.title('Cumulative Returns (simple sum, net of 0.01% TC)\nDL vs Static Benchmarks vs MV/MD — S,B,C,L')
plt.xlabel('Date'); plt.ylabel('Cumulative Return (no compounding)')
plt.grid(True); plt.legend()
plt.tight_layout()
plot_path_cumret = os.path.join(save_dir, 'cumulative_returns_simple_sum_mv_md_TLT.png')
plt.savefig(plot_path_cumret, dpi=200)
plt.close()
print("Saved cumulative return (simple sum) plot:", plot_path_cumret)

# =========================
# 7) NEW: Allocation paths (stacked areas) for DL, MV, MD
# =========================
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axes[0].stackplot(alloc_dl.index, alloc_dl[assets].T.values, labels=assets)
axes[0].set_title('DL — Allocation Weights Over Time')
axes[0].set_ylabel('Weight')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper left', ncol=len(assets))

axes[1].stackplot(alloc_mv.index, alloc_mv[assets].T.values, labels=assets)
axes[1].set_title('MV (max Sharpe, daily; 50-day estimates) — Allocation Weights Over Time')
axes[1].set_ylabel('Weight')
axes[1].grid(True, alpha=0.3)

axes[2].stackplot(alloc_md.index, alloc_md[assets].T.values, labels=assets)
axes[2].set_title('MD (max diversification, daily; 50-day estimates) — Allocation Weights Over Time')
axes[2].set_ylabel('Weight')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
alloc_plot_path = os.path.join(save_dir, 'allocation_paths_DL_MV_MD_TLT.png')
plt.savefig(alloc_plot_path, dpi=200)
plt.close()
print("Saved allocation paths plot:", alloc_plot_path)
