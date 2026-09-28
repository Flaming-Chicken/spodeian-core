/// High-performance quantitative factor calculations and ODE flow kernels.

/// Computes log returns: ln(P_t / P_{t-1}). First element is 0.0.
pub fn calculate_log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![0.0; prices.len()];
    }
    let mut returns = Vec::with_capacity(prices.len());
    returns.push(0.0);

    for i in 1..prices.len() {
        let p_prev = prices[i - 1];
        let p_curr = prices[i];
        if p_prev > 0.0 && p_curr > 0.0 {
            returns.push((p_curr / p_prev).ln());
        } else {
            returns.push(0.0);
        }
    }

    returns
}

/// Computes rolling window sample volatility annualized with sqrt(252).
pub fn calculate_rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.is_empty() || window == 0 {
        return Vec::new();
    }
    let annualization = 252.0_f64.sqrt();
    let mut vol = Vec::with_capacity(returns.len());

    for i in 0..returns.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &returns[start..=i];
        let n = slice.len() as f64;
        if n < 2.0 {
            vol.push(0.0);
            continue;
        }
        let mean = slice.iter().sum::<f64>() / n;
        let var = slice.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        vol.push(var.sqrt() * annualization);
    }

    vol
}

/// Computes Exponential Moving Average (EMA) with span alpha = 2 / (span + 1).
pub fn calculate_ema(series: &[f64], span: usize) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut ema = Vec::with_capacity(series.len());
    let mut prev = series[0];
    ema.push(prev);

    for &val in &series[1..] {
        prev = alpha * val + (1.0 - alpha) * prev;
        ema.push(prev);
    }

    ema
}

/// Computes cumulative Volume-Weighted Average Price (VWAP).
pub fn calculate_vwap(prices: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = prices.len().min(volumes.len());
    if n == 0 {
        return Vec::new();
    }

    let mut vwap = Vec::with_capacity(n);
    let mut cum_pv = 0.0;
    let mut cum_v = 0.0;

    for i in 0..n {
        cum_pv += prices[i] * volumes[i];
        cum_v += volumes[i];
        if cum_v > 0.0 {
            vwap.push(cum_pv / cum_v);
        } else {
            vwap.push(prices[i]);
        }
    }

    vwap
}
