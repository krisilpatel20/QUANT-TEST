import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import io, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

try:
    from fpdf import FPDF
    import xlsxwriter
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

st.set_page_config(page_title="Unified Quant Suite", layout="wide", page_icon="📊")
plt.style.use("ggplot")

st.title("📊 Unified Quant Suite — Thesis + IV Scanner")
st.markdown("""
**Robust Financial Modeling Dashboard** incorporating:
GARCH/EGARCH | Regime Switching | Jump Diffusion | Heston | Kalman Filter | Macro Factors | **Institutional IV Scanner**
""")

if not ARCH_AVAILABLE:
    st.error("⚠️ 'arch' library not installed. Run: pip install arch")

# ==========================================
# HELPER FUNCTIONS & CLASSES
# ==========================================

def format_plot_dates(ax, dates):
    if len(dates) == 0:
        return
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)
    span_days = (dates[-1] - dates[0]).days
    if span_days < 90:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center', fontsize=8)

def highlight_plotly_zones(fig, mask, color, opacity=0.15, row=None, col=None):
    if not isinstance(mask, pd.Series) or not mask.any():
        return
    blocks = (~mask).cumsum()
    for _, group in mask[mask].groupby(blocks[mask]):
        if len(group) > 0:
            x0 = group.index[0]
            x1 = group.index[-1]
            if x0 == x1:
                x1 = x0 + pd.Timedelta(days=1)
            if row is not None and col is not None:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=opacity,
                              layer="below", line_width=0, row=row, col=col)
            else:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=opacity,
                              layer="below", line_width=0)


# ── CALIBRATOR ────────────────────────────────────────────────────────────
class Calibrator:
    @staticmethod
    def calibrate_heston(returns):
        dt = 1/252
        am = arch_model(returns * 100, vol='Garch', p=1, o=0, q=1, dist='Normal')
        res = am.fit(disp='off')
        conditional_vol = res.conditional_volatility / 100
        variance = conditional_vol**2
        variance = variance.values if hasattr(variance, 'values') else variance
        v_curr = variance[:-1]; v_next = variance[1:]
        Y = (v_next - v_curr) / dt; X = v_curr
        A = np.vstack([X, np.ones(len(X))]).T
        beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
        kappa = max(-beta, 0.1)
        theta = max(alpha / kappa if kappa != 0 else np.mean(variance), 0.01)
        residuals = Y - (alpha + beta * X)
        xi = max(np.std(residuals) * np.sqrt(dt) / np.mean(np.sqrt(v_curr)), 0.1)
        rho = np.corrcoef(returns[1:], np.diff(variance))[0, 1]
        mu = np.mean(returns) / dt + 0.5 * np.mean(variance)
        return {'mu': mu, 'kappa': kappa, 'theta': theta, 'xi': xi,
                'rho': rho, 'v0': variance[-1], 'S0': 100.0}


# ── KALMAN FILTERS ────────────────────────────────────────────────────────
class KalmanFilterReg:
    def __init__(self, delta=1e-4, R=1e-3):
        self.delta = delta; self.R = R
        self.trans_cov = delta / (1 - delta) * np.eye(2)

    def run_filter(self, y, x):
        n = len(y)
        state_mean = np.zeros((n, 2)); state_cov = np.zeros((n, 2, 2))
        state_mean[0] = [0, 1]; state_cov[0] = np.eye(2)
        for t in range(1, n):
            pred_state = state_mean[t-1]
            pred_cov = state_cov[t-1] + self.trans_cov
            obs_mat = np.array([[1.0, x[t]]])
            error = y[t] - np.dot(obs_mat, pred_state)
            S = np.dot(np.dot(obs_mat, pred_cov), obs_mat.T) + self.R
            K = np.dot(pred_cov, obs_mat.T) / S
            state_mean[t] = pred_state + K.flatten() * error
            state_cov[t] = pred_cov - np.dot(np.dot(K, obs_mat), pred_cov)
        return state_mean, state_cov


class KalmanFilterTrend:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-3):
        self.Q = process_noise; self.R = measurement_noise

    def filter(self, data):
        n = len(data)
        estimates = np.zeros(n); covariances = np.zeros(n)
        init_window = min(10, max(1, n // 10))
        x = np.mean(data[:init_window])
        P = np.var(data[:init_window]) if init_window > 1 else 1.0
        for t in range(n):
            P_pred = P + self.Q
            K = P_pred / (P_pred + self.R)
            x = x + K * (data[t] - x)
            P = (1 - K) * P_pred
            estimates[t] = x; covariances[t] = P
        return estimates, covariances

    def smooth(self, data):
        n = len(data)
        filtered_means, filtered_covs = self.filter(data)
        smoothed_means = np.zeros(n); smoothed_covs = np.zeros(n)
        smoothed_means[-1] = filtered_means[-1]; smoothed_covs[-1] = filtered_covs[-1]
        for t in range(n - 2, -1, -1):
            P_pred = filtered_covs[t] + self.Q
            J = filtered_covs[t] / P_pred
            smoothed_means[t] = filtered_means[t] + J * (smoothed_means[t+1] - filtered_means[t])
            smoothed_covs[t] = filtered_covs[t] + J**2 * (smoothed_covs[t+1] - P_pred)
        return smoothed_means, smoothed_covs


# ── STOCHASTIC MODELS ─────────────────────────────────────────────────────
def simulate_heston(S0, T, r, kappa, theta, sigma, rho, v0, steps, paths):
    dt = T / steps
    prices = np.zeros((steps + 1, paths)); vols = np.zeros((steps + 1, paths))
    prices[0] = S0; vols[0] = v0
    for t in range(1, steps + 1):
        Z1 = np.random.normal(size=paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=paths)
        v_prev = vols[t-1]
        v_curr = np.abs(v_prev + kappa * (theta - v_prev) * dt +
                        sigma * np.sqrt(np.abs(v_prev)) * np.sqrt(dt) * Z2)
        vols[t] = v_curr
        prices[t] = prices[t-1] + r * prices[t-1] * dt + \
                    np.sqrt(v_curr) * prices[t-1] * np.sqrt(dt) * Z1
    return prices, vols


def merton_jump_diffusion(S0, T, r, sigma, lam, mu_j, sigma_j, steps, paths):
    dt = T / steps
    prices = np.zeros((steps + 1, paths)); prices[0] = S0
    drift = r - 0.5 * sigma**2 - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    for t in range(1, steps + 1):
        z = np.random.normal(size=paths)
        N = np.random.poisson(lam * dt, size=paths)
        J = np.random.normal(mu_j, sigma_j, size=paths) * N
        prices[t] = prices[t-1] * np.exp(drift * dt + sigma * np.sqrt(dt) * z + J)
    return prices


# ── REALIZED VOLATILITY ───────────────────────────────────────────────────
class RealizedVolatility:
    @staticmethod
    def realized_variance(returns):
        return np.sum(returns**2)

    @staticmethod
    def bipower_variation(returns):
        if len(returns) < 2: return 0.0
        abs_rets = np.abs(returns)
        return (np.pi / 2) * np.sum(abs_rets[1:] * abs_rets[:-1])

    @staticmethod
    def jump_component(returns):
        rv = RealizedVolatility.realized_variance(returns)
        bv = RealizedVolatility.bipower_variation(returns)
        jump_var = max(rv - bv, 0)
        jump_ratio = jump_var / rv if rv > 0 else 0.0
        n = len(returns)
        if n < 10:
            return {'jump_ratio': 0.0, 'p_value': 1.0, 'z_score': 0.0}
        z_score = (jump_ratio - 0.05) * np.sqrt(n / 2)
        p_value = 1 - stats.norm.cdf(z_score)
        return {'jump_ratio': jump_ratio, 'p_value': p_value, 'z_score': z_score}


# ── HAWKES VOLATILITY ─────────────────────────────────────────────────────
class HawkesVolatility:
    def __init__(self):
        self.mu = 0.5; self.alpha = 0.5; self.beta = 2.0

    def fit(self, returns):
        vol_proxy = np.abs(returns)
        if len(vol_proxy) < 20: return self
        threshold = np.percentile(vol_proxy, 90)
        events = np.where(vol_proxy > threshold)[0]
        if len(events) < 5: return self

        def neg_ll(params):
            mu_p, alpha_p, beta_p = params
            if mu_p <= 0 or alpha_p < 0 or beta_p <= alpha_p: return 1e9
            t = events; T_end = len(returns)
            R = np.zeros(len(t))
            for i in range(1, len(t)):
                R[i] = np.exp(-beta_p * (t[i] - t[i-1])) * (1 + R[i-1])
            intensities = mu_p + alpha_p * R
            if np.any(intensities <= 0): return 1e9
            term1 = np.sum(np.log(intensities))
            term2 = mu_p * T_end + (alpha_p / beta_p) * np.sum(1 - np.exp(-beta_p * (T_end - t)))
            return -(term1 - term2)

        try:
            res = minimize(neg_ll, [0.1, 0.2, 1.0],
                           bounds=[(1e-4, 2.0), (1e-4, 5.0), (0.1, 10.0)], method='L-BFGS-B')
            self.mu, self.alpha, self.beta = res.x
        except:
            pass
        return self

    def branching_ratio(self):
        return self.alpha / self.beta if self.beta != 0 else 0.0

    def half_life(self):
        return np.log(2) / self.beta if self.beta != 0 else 0.0


# ── PRO REGIME DETECTOR ───────────────────────────────────────────────────
class ProRegimeDetector:
    def __init__(self, prices, log_returns):
        self.prices = prices if isinstance(prices, pd.Series) else pd.Series(prices)
        self.returns = log_returns if isinstance(log_returns, pd.Series) else pd.Series(log_returns)
        self.features = None; self.regimes = {}; self.metrics = {}; self.state_labels = {}

    def _prepare_features(self):
        f1 = self.returns.rolling(window=5).mean().fillna(0)
        vol = self.returns.rolling(window=20).std().bfill()
        v_mean = vol.rolling(252, min_periods=20).mean()
        v_std = vol.rolling(252, min_periods=20).std()
        f2 = ((vol - v_mean) / (v_std + 1e-9)).fillna(0)
        ema = self.prices.ewm(span=20).mean()
        f3 = ((self.prices - ema) / (ema + 1e-9)).fillna(0)
        self.features = np.column_stack([f1.values, f2.values, f3.values])
        return self.features

    def fit(self, n_states=4):
        X = self._prepare_features()
        if not SKLEARN_AVAILABLE:
            self.regimes['states'] = np.zeros(len(X))
            self.regimes['probs'] = np.ones((len(X), 1))
            return
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = GaussianMixture(n_components=n_states, covariance_type='full',
                                random_state=123, max_iter=200)
        model.fit(X_scaled)
        states = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        state_stats = []
        for i in range(n_states):
            mask = (states == i)
            if np.sum(mask) > 0:
                state_stats.append({'id': i,
                                    'ret': np.mean(self.features[mask, 0]),
                                    'vol': np.mean(self.features[mask, 1])})
            else:
                state_stats.append({'id': i, 'ret': -999, 'vol': 999})
        sorted_stats = sorted(state_stats, key=lambda x: x['ret'], reverse=True)
        if n_states == 4:
            bulls = sorted_stats[:2]; bears = sorted_stats[2:]
            bl = min(bulls, key=lambda x: x['vol']); bh = max(bulls, key=lambda x: x['vol'])
            dl = min(bears, key=lambda x: x['vol']); dh = max(bears, key=lambda x: x['vol'])
            self.state_labels = {
                bl['id']: "BULL / QUIET (Conviction)", bh['id']: "BULL / VOLATILE (Exhaustion)",
                dl['id']: "BEAR / QUIET (Distribution)", dh['id']: "BEAR / VOLATILE (Panic/Crisis)"}
        elif n_states == 2:
            self.state_labels = {sorted_stats[0]['id']: "BULL REGIME (Accumulation)",
                                  sorted_stats[1]['id']: "BEAR REGIME (Distribution)"}
        elif n_states == 3:
            self.state_labels = {sorted_stats[0]['id']: "BULL REGIME (Conviction)",
                                  sorted_stats[1]['id']: "NEUTRAL / TRANSITION",
                                  sorted_stats[2]['id']: "BEAR REGIME (Panic)"}
        self.regimes['states'] = states; self.regimes['probs'] = probs
        self.metrics['aic'] = model.aic(X_scaled); self.metrics['bic'] = model.bic(X_scaled)
        self.metrics['n_states'] = n_states

    def fit_optimized(self, state_choices=[2, 3, 4]):
        best_bic = float('inf'); best_n = 4
        for n in state_choices:
            try:
                tmp = ProRegimeDetector(self.prices, self.returns); tmp.fit(n_states=n)
                if tmp.metrics.get('bic', float('inf')) < best_bic:
                    best_bic = tmp.metrics['bic']; best_n = n
            except: continue
        self.fit(n_states=best_n); return best_n

    def get_latest_verdict(self):
        if 'states' not in self.regimes or not self.state_labels:
            return "NEUTRAL", 0.0, "N/A"
        last_state = self.regimes['states'][-1]
        last_prob = np.max(self.regimes['probs'][-1])
        label = self.state_labels.get(last_state, "Unknown")
        if "BULL" in label:
            verdict = "ACCUMULATE / LONG" if "QUIET" in label else "HEDGE / CAUTION"
        elif "BEAR" in label:
            verdict = "DEFENSIVE / SHORT" if "VOLATILE" in label else "REDUCE EXPOSURE"
        else:
            verdict = "NEUTRAL"
        return verdict, last_prob, label


# ── SML ANALYZER ───────────────────────────────────────────────────────
class SMLAnalyzer:
    def __init__(self, ticker_returns, benchmark_returns, rf_annual=0.04):
        self.r_asset = ticker_returns; self.r_bench = benchmark_returns
        self.rf_annual = rf_annual; self.rf_daily = rf_annual / 252

    def calculate_metrics(self, window=90):
        common_idx = self.r_asset.index.intersection(self.r_bench.index)
        y = self.r_asset.loc[common_idx] - self.rf_daily
        x = self.r_bench.loc[common_idx] - self.rf_daily
        df = pd.DataFrame({'asset_ex': y, 'mkt_ex': x}, index=common_idx)
        beta_arr = np.full(len(df), np.nan); alpha_arr = np.full(len(df), np.nan)
        for i in range(window, len(df)):
            ws = df.iloc[i-window:i]
            try:
                model = sm.OLS(ws['asset_ex'], sm.add_constant(ws['mkt_ex'])).fit(
                    cov_type='HAC', cov_kwds={'maxlags': 1})
                alpha_arr[i] = model.params.get('const', np.nan)
                beta_arr[i] = model.params.get('mkt_ex', np.nan)
            except: pass
        df['Beta'] = beta_arr; df['Alpha_Daily'] = alpha_arr
        rolling_mkt_ret_ann = df['mkt_ex'].rolling(window).mean() * 252
        df['SML_Exp_Return'] = self.rf_annual + df['Beta'] * rolling_mkt_ret_ann
        df['Actual_Return_Ann'] = (df['asset_ex'].rolling(window).mean() * 252) + self.rf_annual
        df['Mispricing_Spread'] = df['Actual_Return_Ann'] - df['SML_Exp_Return']
        return df.dropna()


# ── MAD TREND MODES ───────────────────────────────────────────────────────
class MADTrendModes:
    @staticmethod
    def sma(s, l): return s.rolling(window=l).mean()
    @staticmethod
    def ema(s, l): return s.ewm(span=l, adjust=False).mean()
    @staticmethod
    def wma(s, l):
        w = np.arange(1, l+1)
        return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    @staticmethod
    def hma(s, l):
        hl = int(l/2); sl = int(np.sqrt(l))
        combined = 2*MADTrendModes.wma(s, hl) - MADTrendModes.wma(s, l)
        return MADTrendModes.wma(combined.dropna(), sl).reindex_like(s)
    @staticmethod
    def rma(s, l): return s.ewm(alpha=1/l, adjust=False).mean()
    @staticmethod
    def alma(s, l, offset=0.85, sigma=6):
        m = offset*(l-1); si = l/sigma
        w = np.exp(-((np.arange(l)-m)**2)/(2*si*si)); w /= w.sum()
        return s.rolling(l).apply(lambda x: np.dot(x, w), raw=True)
    @staticmethod
    def lsma(s, l):
        def lr(y):
            x = np.arange(len(y)); sl, ic = np.polyfit(x, y, 1)
            return sl*(len(y)-1)+ic
        return s.rolling(l).apply(lr, raw=True)
    @staticmethod
    def ma_switch(s, l, t):
        m = {'SMA': MADTrendModes.sma, 'EMA': MADTrendModes.ema, 'WMA': MADTrendModes.wma,
             'HMA': MADTrendModes.hma, 'RMA': MADTrendModes.rma, 'ALMA': MADTrendModes.alma,
             'LSMA': MADTrendModes.lsma}
        return m.get(t, MADTrendModes.sma)(s, l)
    @staticmethod
    def calculate_mad(series, benchmark, length):
        from numpy.lib.stride_tricks import sliding_window_view
        vals = series.values; bench_vals = benchmark.values
        if len(vals) < length: return pd.Series(np.nan, index=series.index)
        windows = sliding_window_view(vals, length)
        diffs = np.abs(windows - bench_vals[length-1:, np.newaxis])
        res = np.full(len(series), np.nan); res[length-1:] = np.mean(diffs, axis=1)
        return pd.Series(res, index=series.index)
    @staticmethod
    def system_score(series, a, b):
        total = pd.Series(0.0, index=series.index)
        for i in range(a, b+1):
            total += np.sign(series - series.shift(i)).fillna(0)
        return total
    @staticmethod
    def get_signals(df, params):
        src = df['Close']
        mode = params.get('signal_mode', 'Bollinger Bands')
        bb_ma = params.get('bb_ma_type', 'EMA'); bb_len = params.get('bb_len', 25)
        bb_mp = params.get('bb_mult_p', 1.4); bb_mn = params.get('bb_mult_n', 1.0)
        fl_ma = params.get('fl_ma_type', 'ALMA'); fl_len = params.get('fl_len', 10)
        fl_a = params.get('fl_a', 10); fl_b = params.get('fl_b', 60)
        fl_tl = params.get('fl_thresh_l', 23); fl_ts = params.get('fl_thresh_s', 3)
        c_tl = params.get('c_thresh_l', 0.0); c_ts = params.get('c_thresh_s', 0.0)
        avg_bb = MADTrendModes.ma_switch(src, bb_len, bb_ma)
        mad_bb = MADTrendModes.calculate_mad(src, avg_bb, bb_len)
        bb_up = avg_bb + mad_bb*bb_mp; bb_dn = avg_bb - mad_bb*bb_mn
        avg_fl = MADTrendModes.ma_switch(src, fl_len, fl_ma)
        mad_fl = MADTrendModes.calculate_mad(src, avg_fl, fl_len)
        num = MADTrendModes.ma_switch(src*mad_fl, fl_len, fl_ma)
        den = MADTrendModes.ma_switch(mad_fl, fl_len, fl_ma)
        mad_w = num/den
        sys_sc = MADTrendModes.system_score(mad_w, fl_a, fl_b)
        def stateful(lc, sc, idx):
            sig = pd.Series(np.nan, index=idx)
            sig.loc[lc] = 1; sig.loc[sc] = -1
            return sig.ffill().fillna(0)
        bb_sc = stateful((src>bb_up)&(src.shift(1)<=bb_up.shift(1)),
                          (src<bb_dn)&(src.shift(1)>=bb_dn.shift(1)), src.index)
        fl_sc = stateful((sys_sc>fl_tl)&(sys_sc.shift(1)<=fl_tl),
                          (sys_sc<fl_ts)&(sys_sc.shift(1)>=fl_ts), src.index)
        c_sig = (bb_sc+fl_sc)/2
        c_sc = stateful((c_sig>c_tl)&(c_sig.shift(1)<=c_tl),
                         (c_sig<c_ts)&(c_sig.shift(1)>=c_ts), src.index)
        fs = bb_sc if mode=="Bollinger Bands" else (fl_sc if mode=="For Loop" else c_sc)
        return (fs == 1).astype(int)


# ── EHLERS FILTERS ────────────────────────────────────────────────────────
class EhlersFilters:
    @staticmethod
    def super_smoother(prices, period=15):
        a1 = np.exp(-1.414*np.pi/period); b1 = 2*a1*np.cos(1.414*np.pi/period)
        c2 = b1; c3 = -a1*a1; c1 = 1-c2-c3
        filt = np.zeros(len(prices)); pv = prices.values
        for i in range(len(prices)):
            filt[i] = pv[i] if i < 2 else c1*(pv[i]+pv[i-1])/2+c2*filt[i-1]+c3*filt[i-2]
        return pd.Series(filt, index=prices.index)

    @staticmethod
    def simple_decycler(prices, period=60):
        a1 = (np.cos(0.707*2*np.pi/period)+np.sin(0.707*2*np.pi/period)-1)/np.cos(0.707*2*np.pi/period)
        hp = np.zeros(len(prices)); pv = prices.values
        for i in range(len(prices)):
            hp[i] = 0 if i < 2 else ((1-a1/2)**2)*(pv[i]-2*pv[i-1]+pv[i-2]) + \
                    2*(1-a1)*hp[i-1]-((1-a1)**2)*hp[i-2]
        return pd.Series(pv-hp, index=prices.index)


# ── VOL-TARGETED SIZING ───────────────────────────────────────────────────
def vol_targeted_signal(raw_signal, garch_res, target_vol_annual=0.15):
    """Converts binary 0/1 signal to fractional 0.0-1.0 using vol targeting."""
    if garch_res is None: return raw_signal.astype(float)
    cond_vol = garch_res.conditional_volatility / 100.0 * np.sqrt(252)
    common = raw_signal.index.intersection(cond_vol.index)
    if len(common) == 0: return raw_signal.astype(float)
    sig = raw_signal.loc[common].astype(float)
    vol = cond_vol.loc[common]
    size = np.minimum(1.0, target_vol_annual / (vol + 1e-9))
    return (sig * size).fillna(0.0)


# ── IMPROVED HURST SIGNAL ────────────────────────────────────────────────
def rolling_hurst(prices, window=100, max_lag=20):
    log_prices = np.log(prices)
    def hurst_val(x):
        lags = range(2, max_lag)
        tau = [max(np.std(x[lag:]-x[:-lag]), 1e-8) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return log_prices.rolling(window).apply(hurst_val, raw=True)


def hurst_confirmed_signal(prices_bt, hurst_window=100, trend_thresh=0.55,
                            mr_thresh=0.45, confirm_bars=5):
    """
    Dead zone 0.45-0.55 + 5-bar confirmation to eliminate whipsaw.
    H>0.55 confirmed -> momentum, H<0.45 confirmed -> mean reversion, else -> cash.
    """
    hurst_series = rolling_hurst(prices_bt, window=hurst_window)

    def confirmed(h, threshold, above=True):
        raw = (h > threshold).astype(int) if above else (h < threshold).astype(int)
        return raw.rolling(confirm_bars).min().fillna(0).astype(int)

    is_trending = confirmed(hurst_series, trend_thresh, above=True)
    is_mr = confirmed(hurst_series, mr_thresh, above=False)

    ema_fast = prices_bt.ewm(span=20, adjust=False).mean()
    ema_slow = prices_bt.ewm(span=50, adjust=False).mean()
    trend_sig = (ema_fast > ema_slow).astype(int)

    bb_ma = prices_bt.rolling(20).mean()
    bb_std = prices_bt.rolling(20).std()
    bb_lower = bb_ma - 2*bb_std
    mr_raw = pd.Series(np.nan, index=prices_bt.index)
    mr_raw.loc[prices_bt < bb_lower] = 1
    mr_raw.loc[prices_bt > bb_ma] = 0
    mr_sig = mr_raw.ffill().fillna(0)

    signals = pd.Series(0, index=prices_bt.index)
    signals[is_trending == 1] = trend_sig[is_trending == 1]
    signals[is_mr == 1] = mr_sig[is_mr == 1]
    return signals.fillna(0), hurst_series, is_trending, is_mr


# ── GARCH VOL FILTER ─────────────────────────────────────────────────────
def garch_vol_filter_signal(prices_bt, returns_bt, base_signal, target_vol_pct=15.0):
    """
    Replaces VIX proxy with stock's own GARCH vol state.
    Returns fractional signal sized by vol target instead of binary on/off.
    """
    if not ARCH_AVAILABLE: return base_signal.astype(float)
    try:
        rs = (returns_bt * 100).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rs) < 30: return base_signal.astype(float)
        am = arch_model(rs, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off')
        return vol_targeted_signal(base_signal, res, target_vol_annual=target_vol_pct/100.0)
    except:
        return base_signal.astype(float)


# ── BACKTEST ENGINE ───────────────────────────────────────────────────────
class BacktestEngine:
    @staticmethod
    def run_strategy(prices, signals, initial_capital=10000.0,
                     trailing_stop_pct=0.0, stop_loss_pct=0.0):
        common_idx = prices.index.intersection(signals.index)
        prices = prices.loc[common_idx]; signals = signals.loc[common_idx]
        returns = prices.pct_change().fillna(0)
        equity_curve = [initial_capital]
        trades = []; position = 0; entry_price = 0; entry_date = None
        max_price = 0; cash = initial_capital; holdings = 0
        cooldown = 0; prev_sig = 0.0

        for date, price, signal in zip(prices.index, prices, signals):
            signal = float(signal)
            if cooldown > 0:
                cooldown -= 1
                if signal == 0: cooldown = 0

            if position == 1:
                stop_out = False; smsg = ""
                if stop_loss_pct > 0 and price <= entry_price*(1-stop_loss_pct):
                    stop_out = True; smsg = 'Stop Loss'
                if not stop_out and trailing_stop_pct > 0:
                    max_price = max(max_price, price)
                    if price <= max_price*(1-trailing_stop_pct):
                        stop_out = True; smsg = 'Trailing Stop'
                if stop_out:
                    position = 0; cash = holdings*price; holdings = 0
                    pnl = (price-entry_price)/entry_price
                    trades.append({'Side':'Long','Entry Date':entry_date,'Exit Date':date,
                                   'Buy Price':entry_price,'Sell Price':price,
                                   'PnL (%)':pnl*100,'Status':smsg})
                    equity_curve.append(cash); cooldown = 5; prev_sig = 0.0; continue

            if position == 0 and signal > 0 and cooldown == 0:
                position = 1; entry_price = price; entry_date = date; max_price = price
                invest = cash * signal; holdings = invest/price; cash -= invest
            elif position == 1 and signal == 0:
                position = 0; cash += holdings*price
                pnl = (price-entry_price)/entry_price; holdings = 0
                trades.append({'Side':'Long','Entry Date':entry_date,'Exit Date':date,
                               'Buy Price':entry_price,'Sell Price':price,
                               'PnL (%)':pnl*100,'Status':'Closed'})
            elif position == 1 and signal != prev_sig and signal > 0:
                total = cash + holdings*price
                holdings = (total*signal)/price; cash = total - holdings*price
            prev_sig = signal
            equity_curve.append((cash+holdings*price) if position==1 else cash)

        if position == 1:
            cp = prices.iloc[-1]; pnl = (cp-entry_price)/entry_price
            trades.append({'Side':'Long','Entry Date':entry_date,'Exit Date':None,
                           'Buy Price':entry_price,'Sell Price':cp,
                           'PnL (%)':pnl*100,'Status':'Open'})
            equity_curve[-1] = cash+holdings*cp

        eq = pd.Series(equity_curve[1:], index=prices.index)
        bm = initial_capital*(1+returns).cumprod()
        return {'equity_curve':eq,'benchmark_curve':bm,
                'trades':pd.DataFrame(trades),'returns':eq.pct_change().fillna(0)}

    @staticmethod
    def calculate_metrics(returns, risk_free_rate=0.0):
        if len(returns) < 2: return {}
        excess = returns - risk_free_rate/252
        sharpe = np.sqrt(252)*excess.mean()/(returns.std()+1e-9)
        down = returns[returns<0]
        sortino = np.sqrt(252)*excess.mean()/(down.std()+1e-9)
        cum = (1+returns).cumprod()
        max_dd = ((cum-cum.cummax())/cum.cummax()).min()
        n_years = len(returns)/252
        cagr = ((1+returns).prod()**(1/n_years)-1) if n_years>0 else 0
        return {'Sharpe Ratio':sharpe,'Sortino Ratio':sortino,'Max Drawdown':max_dd,'CAGR':cagr}

# ── REGIME MODEL FIT (CACHED) ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fit_regime_model(model_data, n_regimes, switch_vol, switch_trend, search_reps=20):
    if hasattr(model_data, 'values'):
        clean = model_data.values.flatten().astype(float); idx = model_data.index
    else:
        clean = np.array(model_data).flatten().astype(float); idx = pd.RangeIndex(len(clean))
    if np.any(np.isnan(clean)) or np.any(np.isinf(clean)):
        st.error("Data contains NaN/Inf. Cannot fit model."); return None
    if np.std(clean) < 1e-9:
        st.error("Data is constant. Cannot fit model."); return None
    endog = pd.Series(clean, index=idx)
    try:
        mod = MarkovRegression(endog, k_regimes=n_regimes, trend='c',
                               switching_variance=switch_vol, switching_trend=switch_trend)
        res = mod.fit(search_reps=search_reps, disp=False)
        if isinstance(res.params, np.ndarray):
            names = res.model.param_names
            res.params = pd.Series(res.params, index=names)
            res.bse = pd.Series(res.bse, index=names)
            res.pvalues = pd.Series(res.pvalues, index=names)
        return res
    except Exception as e:
        st.error(f"Fit failed: {e}"); return None


# ── MASTER SIGNAL ENGINE ───────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_master_signal(ticker, df, n_regimes=4, freq='Daily', opt_goal='Robustness (BIC)',
                       stability=0, switch_vol=True, switch_trend=True, engine='Markov',
                       initial_cap=10000.0, trailing_stop=0.0, stop_loss=0.0):
    try:
        df = df.replace([np.inf,-np.inf], np.nan).dropna()
        if stability > 0:
            for col in ['Returns','Log_Returns','Close']:
                if col in df.columns:
                    df[col] = df[col].ewm(span=stability, adjust=False).mean()
        if freq == 'Weekly':
            df = df.resample('W').last().replace([np.inf,-np.inf], np.nan).dropna()
            df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
            df['Returns'] = df['Close'].pct_change()
            df = df.replace([np.inf,-np.inf], np.nan).dropna()
        if len(df) < 15: return None

        if engine == 'Markov':
            if n_regimes == 'Auto':
                best_n=4; best_sc=-float('inf') if opt_goal=='Performance (PnL)' else float('inf')
                best_r=None
                for n in [2,3,4]:
                    try:
                        r = fit_regime_model(df['Returns']*100, n, switch_vol, switch_trend, search_reps=5)
                        if r:
                            sc = r.bic if opt_goal != 'Performance (PnL)' else 0
                            if (opt_goal=='Robustness (BIC)' and sc < best_sc) or \
                               (opt_goal=='Performance (PnL)' and sc > best_sc):
                                best_sc=sc; best_n=n; best_r=r
                    except: continue
                res_markov = best_r
            else:
                res_markov = fit_regime_model(df['Returns']*100, int(n_regimes), switch_vol, switch_trend)
            if not res_markov: return None
            p_df = res_markov.filtered_marginal_probabilities
            n_st = res_markov.k_regimes
            r_means = []
            for i in range(n_st):
                m = res_markov.params[f'const[{i}]'] if f'const[{i}]' in res_markov.params \
                    else res_markov.params.get('const', 0.0)
                r_means.append((i, m))
            bull_idx = sorted(r_means, key=lambda x: x[1], reverse=True)[0][0]
            bear_idx = sorted(r_means, key=lambda x: x[1])[0][0]
            curr = p_df.iloc[-1].idxmax(); prob = p_df.iloc[-1].max()
            if curr == bull_idx: sig, lbl = "LONG", "BULL"
            elif curr == bear_idx: sig, lbl = "SHORT", "BEAR"
            else: sig, lbl = "CASH", "NEUTRAL"
            regime_data = {'label': lbl, 'confidence': prob, 'n_states': n_st}
            p_detector = None
        else:
            p_detector = ProRegimeDetector(df['Close'], df['Log_Returns'])
            if n_regimes == 'Auto': p_detector.fit_optimized()
            else: p_detector.fit(n_states=int(n_regimes))
            sig, prob, lbl = p_detector.get_latest_verdict()
            regime_data = {'label': lbl, 'confidence': prob,
                           'n_states': p_detector.metrics.get('n_states', 4)}

        kf = KalmanFilterTrend(process_noise=1e-4, measurement_noise=1e-2)
        trend_est, _ = kf.filter(df['Close'].values)
        last_price = df['Close'].iloc[-1]; last_trend = trend_est[-1]
        trend_diff = (last_price - last_trend) / (last_trend + 1e-9)

        rs = (df['Returns']*100).replace([np.inf,-np.inf], np.nan).dropna()
        if len(rs) < 15: return None
        am = arch_model(rs, vol='Garch', p=1, q=1, dist='Normal')
        garch_res = am.fit(disp='off')
        curr_vol = garch_res.conditional_volatility.iloc[-1]
        avg_vol = garch_res.conditional_volatility.mean()
        vol_state = "HIGH" if curr_vol > avg_vol*1.2 else "LOW" if curr_vol < avg_vol*0.8 else "NORMAL"

        jump_res = RealizedVolatility.jump_component(df['Returns'].values)
        jump_detected = jump_res['p_value'] < 0.05

        score = 0
        if "LONG" in sig: score += 2
        if "SHORT" in sig: score -= 2
        if trend_diff > 0.01: score += 1
        if trend_diff < -0.01: score -= 1
        if vol_state == "LOW": score += 1
        if vol_state == "HIGH": score -= 1
        if jump_detected: score -= 1

        return {'regime_sig': sig, 'regime_label': lbl, 'regime_data': regime_data,
                'regime_prob': prob, 'pro_detector': p_detector, 'trend_diff': trend_diff,
                'vol_state': vol_state, 'curr_vol': curr_vol, 'jump_detected': jump_detected,
                'sentiment_score': score, 'garch_res': garch_res}
    except Exception as e:
        st.error(f"Error in Decision Engine for {ticker}: {e}"); return None


# ── DATA LOADING ───────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(ticker, start, end, interval='1d'):
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(ticker, axis=1, level=1, drop_level=True)
            except: df.columns = df.columns.droplevel(1)
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        if 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        return df.replace([np.inf,-np.inf], np.nan).dropna()
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}"); return None


@st.cache_data(ttl=3600)
def load_fred_data(series_id):
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url)
        df['DATE'] = pd.to_datetime(df['DATE']); df.set_index('DATE', inplace=True)
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        return df
    except: return None


@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    except:
        return ["AAPL","MSFT","AMZN","GOOG","NVDA","META","TSLA","BRK-B","UNH","JNJ"]


@st.cache_data(ttl=86400)
def get_nasdaq100_tickers():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        for t in tables:
            if 'Ticker' in t.columns: return t['Ticker'].tolist()
        return tables[4].iloc[:,1].tolist()
    except:
        return ["AAPL","MSFT","AMZN","GOOG","NVDA","META","TSLA","PEP","AVGO","COST"]


@st.cache_data(ttl=86400)
def get_total_us_stocks():
    import requests
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, headers={'User-Agent':'QuantApp/1.0 (admin@q.local)'}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            tickers = sorted(list(set([v['ticker'] for v in data.values()
                                       if len(str(v['ticker'])) <= 5 and '-' not in str(v['ticker'])])))
            if len(tickers) > 5000: return tickers
    except: pass
    return get_sp500_tickers()


def get_market_cap(ticker):
    try: return yf.Ticker(ticker).info.get('marketCap', 0)
    except: return 0


def get_analyst_target(ticker):
    try:
        info = yf.Ticker(ticker).info
        target = info.get('targetMeanPrice')
        current = info.get('currentPrice') or info.get('previousClose')
        if target and current: return target, np.log(target/current)
        return None, None
    except: return None, None


def calculate_beta(ticker_returns, benchmark_ticker='SPY', lookback_years=2):
    try:
        end = datetime.now(); start = end - timedelta(days=lookback_years*365)
        bench = yf.download(benchmark_ticker, start=start, end=end, progress=False)
        if isinstance(bench.columns, pd.MultiIndex): bench.columns = bench.columns.droplevel(1)
        if 'Close' not in bench.columns and 'Adj Close' in bench.columns:
            bench['Close'] = bench['Adj Close']
        bench_ret = bench['Close'].pct_change().dropna()
        common = ticker_returns.index.intersection(bench_ret.index)
        if len(common) < 30: return 1.0
        y = ticker_returns.loc[common]; x = bench_ret.loc[common]
        return np.cov(y, x)[0,1] / np.var(x)
    except: return 1.0


# ── REPORT GENERATOR ───────────────────────────────────────────────────────
class ReportGenerator:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker; self.start_date = start_date
        self.end_date = end_date; self.data_store = {}; self.plots = {}

    def add_data(self, key, df_or_dict): self.data_store[key] = df_or_dict

    def add_plot(self, key, fig):
        buf = io.BytesIO()
        if hasattr(fig, 'savefig'): fig.savefig(buf, format='png', bbox_inches='tight')
        elif hasattr(fig, 'write_image'):
            try: fig.write_image(buf, format='png', engine='kaleido')
            except: return
        else: return
        buf.seek(0); self.plots[key] = buf

    def generate_excel(self):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for key, data in self.data_store.items():
                sn = "".join([c for c in key if c.isalnum() or c in (" ","_")])[:31]
                if isinstance(data, pd.DataFrame): data.to_excel(writer, sheet_name=sn)
                elif isinstance(data, dict):
                    pd.DataFrame(list(data.items()), columns=['Metric','Value']).to_excel(
                        writer, sheet_name=sn, index=False)
        return output.getvalue()

    def generate_pdf(self):
        pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
        pdf.set_font("Arial",'B',24); pdf.set_text_color(44,62,80)
        pdf.cell(0,20,f"Unified Quant Analysis Report",ln=True,align='C')
        pdf.set_font("Arial",'B',16); pdf.cell(0,10,f"Asset: {self.ticker}",ln=True,align='C')
        pdf.set_font("Arial",size=10); pdf.set_text_color(100,100,100)
        pdf.cell(0,10,f"Period: {self.start_date} to {self.end_date}",ln=True,align='C')
        pdf.cell(0,10,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",ln=True,align='C')
        pdf.ln(10); printed = set()
        for key, data in self.data_store.items():
            pdf.set_font("Arial",'B',14); pdf.set_text_color(31,119,180)
            pdf.cell(0,10,f"SECTION: {key}",ln=True)
            pdf.set_font("Arial",size=10); pdf.set_text_color(0,0,0)
            if isinstance(data, dict):
                for k,v in data.items():
                    vs = f"{v:.4f}" if isinstance(v,(float,np.float64,np.float32)) else str(v)
                    pdf.cell(70,6,f"{k}:",border=0); pdf.cell(0,6,vs,ln=True,border=0)
            elif isinstance(data, pd.DataFrame):
                pdf.cell(0,7,f"Table: {len(data)} rows (see Excel)",ln=True)
            if key in self.plots:
                pdf.ln(2); pdf.image(self.plots[key],x=15,w=180); printed.add(key); pdf.ln(5)
            pdf.ln(8)
            if pdf.get_y() > 230: pdf.add_page()
        for key in [k for k in self.plots if k not in printed]:
            if pdf.get_y() > 100: pdf.add_page()
            pdf.set_font("Arial",'B',12); pdf.cell(0,10,key,ln=True)
            pdf.image(self.plots[key],x=15,w=180); pdf.ln(10)
            if pdf.get_y() > 230: pdf.add_page()
        raw = pdf.output()
        return bytes(raw) if isinstance(raw, bytearray) else raw


# ── FED DATA ────────────────────────────────────────────────────────────
FED_ASSETS = {
    "WGCAL":"Gold Certificate Account","SDRACL":"SDR Certificate Account",
    "WCOINL":"Coin","WSHONBLL":"Treasury Bills","WSHONBNL":"Treasury Notes and Bonds",
    "WSHONBIIL":"Treasury Tips","WSHOMCB":"Mortgage-Backed Securities",
    "WUDSHO":"Unamortized Premiums","WLCFOCEL":"Other Credit Extensions","WOTHAL":"Other Assets"
}
FED_LIABILITIES = {
    "WCURCIR":"Currency in Circulation","WDTPGCAS":"Treasury General Account (TGA)",
    "RRPONTSYD":"Overnight Reverse Repo (RRP)","WLRRAL":"Reverse Repos (Total)",
    "WLFN":"Federal Reserve Notes","WDFOL":"Foreign Official Deposits",
    "WDLTCL":"Term Deposits"
}


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Thesis Parameters")
    market_region = st.selectbox("Market Region", ["US Market (USD)","Indian Market (INR)","Futures / Commodities (USD)"])
    if market_region == "Indian Market (INR)":
        CURRENCY="₹"; BENCHMARK="^NSEI"; DEFAULT_RF=7.0; SUFFIX=".NS"
    elif market_region == "Futures / Commodities (USD)":
        CURRENCY="$"; BENCHMARK="GC=F"; DEFAULT_RF=4.0; SUFFIX=""
    else:
        CURRENCY="$"; BENCHMARK="SPY"; DEFAULT_RF=4.0; SUFFIX=""

    col_t1,col_t2 = st.columns(2)
    with col_t1:
        raw_ticker = st.text_input("Main Ticker","RELIANCE" if market_region=="Indian Market (INR)" else "AAPL").upper()
    with col_t2:
        raw_pair = st.text_input("Pair Ticker","").upper()

    TICKER = raw_ticker+SUFFIX if (SUFFIX and not raw_ticker.endswith(SUFFIX)) else raw_ticker
    PAIR_TICKER = raw_pair+SUFFIX if (SUFFIX and raw_pair and not raw_pair.endswith(SUFFIX)) else raw_pair
    st.caption(f"Active Ticker: **{TICKER}**")

    start_date = st.date_input("Start Date", datetime.now()-timedelta(days=365))
    end_date = st.date_input("End Date", datetime.now())
    rf_rate = st.number_input("Risk Free Rate (%)",0.0,20.0,DEFAULT_RF)/100
    st.info(f"Benchmark: {BENCHMARK} | Currency: {CURRENCY}")
    st.divider()

    st.header("🔬 Model Config")
    regime_mode = st.selectbox("Regime Mode",
        ["Fixed: 4 States (Inst.)","Fixed: 2 States (Bull/Bear)","Fixed: 3 States","Auto: Best Fit (BIC)"])
    regime_val_map = {"Fixed: 4 States (Inst.)":4,"Fixed: 2 States (Bull/Bear)":2,
                      "Fixed: 3 States":3,"Auto: Best Fit (BIC)":"Auto"}
    regime_param = regime_val_map[regime_mode]

    with st.expander("Advanced Model Sync"):
        reg_engine = st.selectbox("Engine",["Markov (High Accuracy)","GMM (Fast)"])
        reg_engine_param = "Markov" if "Markov" in reg_engine else "GMM"
        reg_stability = st.slider("Signal Stability",0,10,4)
        reg_opt_goal = st.selectbox("Optimization Goal",["Robustness (BIC)","Performance (PnL)"])
        reg_switch_vol = st.toggle("Switching Volatility",value=True)
        reg_switch_trend = st.toggle("Switching Mean",value=True)
        initial_cap = st.number_input("Initial Capital",1000,1000000,10000)
        use_ts = st.toggle("Trailing Stop Loss",value=False)
        trailing_stop = st.slider("Trailing Stop (%)",0.0,20.0,5.0,step=0.5)/100 if use_ts else 0.0
        use_sl = st.toggle("Hard Stop Loss",value=True)
        stop_loss = st.slider("Hard Stop (%)",0.0,30.0,8.0,step=0.5)/100 if use_sl else 0.0

    st.divider()
    st.header("⚡ Live Mode")
    live_mode = st.toggle("Enable Live Data",value=False)
    if live_mode:
        data_interval = st.selectbox("Interval",["1m","5m","15m","60m"],index=1)
        if st.button("🔄 Refresh",use_container_width=True):
            st.cache_data.clear(); st.rerun()
    else:
        data_interval = '1d'

    st.divider()
    st.header("📥 Export")
    if not EXPORT_AVAILABLE:
        st.error("Missing: fpdf2, xlsxwriter"); st.info("pip install fpdf2 xlsxwriter")
    else:
        if 'report_gen' not in st.session_state: st.session_state.report_gen = None
        if st.session_state.report_gen:
            c1,c2 = st.columns(2)
            with c1:
                try:
                    raw = st.session_state.report_gen.generate_pdf()
                    st.download_button("📄 PDF",bytes(raw) if isinstance(raw,bytearray) else raw,
                        f"Quant_{TICKER}.pdf","application/pdf")
                except Exception as e: st.error(f"PDF: {e}")
            with c2:
                try:
                    st.download_button("📊 Excel",st.session_state.report_gen.generate_excel(),
                        f"Quant_{TICKER}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e: st.error(f"Excel: {e}")
        else: st.caption("Run analysis to enable exports.")


# ==========================================
# DATA LOADING
# ==========================================
now_r = datetime.now().replace(second=0, microsecond=0)
if live_mode:
    lb = 7 if data_interval=='1m' else 30
    start_date = now_r - timedelta(days=lb)

df = load_data(TICKER, start_date, end_date, interval=data_interval)

if df is not None:
    st.success(f"✅ Loaded {len(df)} bars for {TICKER}")
    st.write(f"**Data Range:** {df.index[0].date()} → {df.index[-1].date()}")
else:
    st.error(f"❌ Failed to load {TICKER}")
    st.stop()


# ==========================================
# MAIN ANALYSIS
# ==========================================
master_result = get_master_signal(
    TICKER, df, n_regimes=regime_param, freq='Daily',
    opt_goal=reg_opt_goal, stability=reg_stability,
    switch_vol=reg_switch_vol, switch_trend=reg_switch_trend,
    engine=reg_engine_param, initial_cap=initial_cap,
    trailing_stop=trailing_stop, stop_loss=stop_loss
)

if master_result:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Regime Signal", master_result['regime_sig'],
                  delta=f"{master_result['regime_prob']:.1%} Confidence")
    with col2:
        st.metric("Regime Label", master_result['regime_label'])
    with col3:
        st.metric("Vol State", master_result['vol_state'])
    with col4:
        st.metric("Sentiment Score", f"{master_result['sentiment_score']:.0f}", 
                  delta="Positive" if master_result['sentiment_score'] > 0 else "Negative")

st.divider()
st.write("**Analysis Complete** - Dashboard Ready for Export")
