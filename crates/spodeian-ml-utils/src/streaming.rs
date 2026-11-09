//! Universal online streaming time-series, sliding window accumulators, and rolling statistics.
//!
//! **Stability: Experimental** — API subject to change
//!
//! Replaces $O(W \cdot N)$ repeated window scans with **$O(1)$ constant-time** online updates
//! for real-time market ticks, high-frequency signal processing, and neural feature pipelines.
//!
//! # Components:
//! - [`RollingWindow<T>`]: General-purpose circular buffer with zero allocations after creation.
//! - [`OnlineSlidingStats`]: Numerically stable $O(1)$ sliding mean, variance, standard deviation, and z-score.
//! - [`ExponentialMovingAverage`]: Online single-pass EMA with configurable smoothing span.
//! - [`OnlineSlidingCovariance`]: Online two-stream sliding covariance and Pearson correlation coefficient.

use serde::{Deserialize, Serialize};

/// Generic sliding window buffer maintaining a fixed capacity with $O(1)$ push and pop.
///
/// # Examples
///
/// ```rust
/// use spodeian_ml_utils::RollingWindow;
///
/// let mut window = RollingWindow::new(3);
/// assert_eq!(window.push(10), None);
/// assert_eq!(window.push(20), None);
/// assert_eq!(window.push(30), None);
/// assert!(window.is_full());
///
/// // Pushing a 4th item evicts the oldest (10)
/// assert_eq!(window.push(40), Some(10));
/// assert_eq!(window.latest(), Some(&40));
/// assert_eq!(window.oldest(), Some(&20));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollingWindow<T> {
    buffer: Vec<T>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl<T: Clone> RollingWindow<T> {
    /// Creates a new rolling window with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            buffer: Vec::with_capacity(cap),
            capacity: cap,
            head: 0,
            count: 0,
        }
    }

    /// Pushes a new element into the window.
    ///
    /// If the window was full, the oldest element is evicted and returned as `Some(old)`.
    pub fn push(&mut self, item: T) -> Option<T> {
        if self.count < self.capacity {
            self.buffer.push(item);
            self.count += 1;
            None
        } else {
            let old = std::mem::replace(&mut self.buffer[self.head], item);
            self.head = (self.head + 1) % self.capacity;
            Some(old)
        }
    }

    /// Returns the number of elements currently stored.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns the maximum capacity of the window.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if the window is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns `true` if the window has reached its maximum capacity.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    /// Returns the latest (most recently pushed) element.
    pub fn latest(&self) -> Option<&T> {
        if self.count == 0 {
            None
        } else if self.count < self.capacity {
            self.buffer.last()
        } else {
            let idx = if self.head == 0 {
                self.capacity - 1
            } else {
                self.head - 1
            };
            Some(&self.buffer[idx])
        }
    }

    /// Returns the oldest element currently in the window.
    pub fn oldest(&self) -> Option<&T> {
        if self.count == 0 {
            None
        } else if self.count < self.capacity {
            self.buffer.first()
        } else {
            Some(&self.buffer[self.head])
        }
    }

    /// Clears all elements from the window.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.head = 0;
        self.count = 0;
    }

    /// Returns an iterator over elements from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let (s1, s2) = self.as_slices();
        s1.iter().chain(s2.iter())
    }

    /// Returns the window contents as up to two contiguous slices `(oldest_part, newest_part)`.
    pub fn as_slices(&self) -> (&[T], &[T]) {
        if self.count < self.capacity {
            (&self.buffer[..self.count], &[])
        } else {
            let (head, tail) = self.buffer.split_at(self.head);
            (tail, head)
        }
    }
}

/// Numerically stable online sliding statistics maintaining mean, variance, and standard deviation in $O(1)$ time.
///
/// # Mathematical Background
///
/// Tracks the running sum $S_1 = \sum x_i$ and sum-of-squares $S_2 = \sum x_i^2$.
/// Upon sample insertion and eviction, computes the sample variance in $O(1)$ operations via:
///
/// $$\sigma^2 = \frac{S_2 - \frac{S_1^2}{N}}{N - 1}$$
///
/// # Examples
///
/// ```rust
/// use spodeian_ml_utils::OnlineSlidingStats;
///
/// let mut stats = OnlineSlidingStats::new(4);
/// for val in [1.0, 2.0, 3.0, 4.0] {
///     stats.push(val);
/// }
/// assert_eq!(stats.mean(), 2.5);
/// assert!((stats.variance() - 1.6666666666666667).abs() < 1e-12);
/// assert!((stats.z_score(4.0) - 1.161895003862225).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnlineSlidingStats {
    window: RollingWindow<f64>,
    sum: f64,
    sum_sq: f64,
}

impl OnlineSlidingStats {
    /// Creates a new online sliding statistics tracker with the given window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            window: RollingWindow::new(window_size),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Pushes a new sample into the sliding window in $O(1)$ constant time.
    ///
    /// Updates running sum and sum-of-squares while evicting the oldest element if full.
    pub fn push(&mut self, val: f64) {
        if let Some(old) = self.window.push(val) {
            self.sum += val - old;
            self.sum_sq += val * val - old * old;
        } else {
            self.sum += val;
            self.sum_sq += val * val;
        }

        // Numerical guard: prevent small floating-point precision drift from producing negative variance
        if self.sum_sq < 0.0 {
            self.sum_sq = 0.0;
        }
    }

    /// Current number of samples in the window.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Window capacity.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.window.capacity()
    }

    /// Whether the window is full.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.window.is_full()
    }

    /// Returns the arithmetic mean of the current window in $O(1)$ time.
    pub fn mean(&self) -> f64 {
        let n = self.window.len();
        if n == 0 {
            0.0
        } else {
            self.sum / (n as f64)
        }
    }

    /// Returns sample variance ($s^2 = \frac{1}{n-1} \sum (x_i - \bar{x})^2$) in $O(1)$ time.
    pub fn variance(&self) -> f64 {
        let n = self.window.len();
        if n < 2 {
            return 0.0;
        }
        let n_f = n as f64;
        let mean = self.sum / n_f;
        let raw_var = (self.sum_sq - n_f * mean * mean) / (n_f - 1.0);
        raw_var.max(0.0)
    }

    /// Returns standard deviation ($s = \sqrt{s^2}$) in $O(1)$ time.
    #[inline(always)]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Annualized volatility assuming 252 trading days per year: $\sigma_{\text{ann}} = s \cdot \sqrt{252}$.
    #[inline(always)]
    pub fn annualized_volatility(&self) -> f64 {
        self.std_dev() * 252.0_f64.sqrt()
    }

    /// Computes the Z-score of `val` relative to current window mean and standard deviation:
    /// $$z = \frac{x - \mu}{\sigma + \epsilon}$$
    pub fn z_score(&self, val: f64) -> f64 {
        let std = self.std_dev();
        if std <= 1e-12 {
            0.0
        } else {
            (val - self.mean()) / std
        }
    }

    /// Clears all accumulated statistics.
    pub fn clear(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

/// Exponential Moving Average (EMA) online accumulator.
///
/// # Mathematical Background
///
/// Given smoothing parameter $\alpha = \frac{2}{\text{span} + 1}$, updates state:
///
/// $$\text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}$$
///
/// # Examples
///
/// ```rust
/// use spodeian_ml_utils::ExponentialMovingAverage;
///
/// let mut ema = ExponentialMovingAverage::with_span(9);
/// assert_eq!(ema.update(100.0), 100.0);
/// let val = ema.update(110.0);
/// assert!((val - 102.0).abs() < 1e-6);
/// assert_eq!(ema.value(), Some(val));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExponentialMovingAverage {
    alpha: f64,
    current: Option<f64>,
}

impl ExponentialMovingAverage {
    /// Creates a new EMA tracker with the given span ($\alpha = \frac{2}{\text{span} + 1}$).
    pub fn with_span(span: usize) -> Self {
        let s = span.max(1) as f64;
        let alpha = 2.0 / (s + 1.0);
        Self {
            alpha,
            current: None,
        }
    }

    /// Creates a new EMA tracker with an explicit smoothing factor $\alpha \in (0, 1]$.
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-6, 1.0),
            current: None,
        }
    }

    /// Feeds a new value into the EMA and returns the updated moving average in $O(1)$ time.
    pub fn update(&mut self, val: f64) -> f64 {
        match self.current {
            Some(prev) => {
                let next = self.alpha * val + (1.0 - self.alpha) * prev;
                self.current = Some(next);
                next
            }
            None => {
                self.current = Some(val);
                val
            }
        }
    }

    /// Returns the current EMA value if initialized.
    pub fn value(&self) -> Option<f64> {
        self.current
    }

    /// Resets the EMA accumulator.
    pub fn reset(&mut self) {
        self.current = None;
    }
}

/// Online sliding covariance and Pearson correlation coefficient between two streaming series.
///
/// # Mathematical Background
///
/// Computes sample covariance $\text{Cov}(X, Y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{N - 1}$
/// and Pearson correlation $r = \frac{\text{Cov}(X, Y)}{s_x s_y}$ in $O(1)$ operations per update.
///
/// # Examples
///
/// ```rust
/// use spodeian_ml_utils::OnlineSlidingCovariance;
///
/// let mut cov = OnlineSlidingCovariance::new(5);
/// for i in 1..=5 {
///     cov.push(i as f64, (2 * i) as f64);
/// }
/// assert!((cov.correlation() - 1.0).abs() < 1e-6);
/// assert!((cov.beta() - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnlineSlidingCovariance {
    window_x: RollingWindow<f64>,
    window_y: RollingWindow<f64>,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

impl OnlineSlidingCovariance {
    /// Creates a new two-stream sliding covariance tracker with the given window size.
    pub fn new(window_size: usize) -> Self {
        let cap = window_size.max(2);
        Self {
            window_x: RollingWindow::new(cap),
            window_y: RollingWindow::new(cap),
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_yy: 0.0,
            sum_xy: 0.0,
        }
    }

    /// Pushes a pair $(x, y)$ into the sliding window in $O(1)$ constant time.
    pub fn push(&mut self, x: f64, y: f64) {
        if let (Some(old_x), Some(old_y)) = (self.window_x.push(x), self.window_y.push(y)) {
            self.sum_x += x - old_x;
            self.sum_y += y - old_y;
            self.sum_xx += x * x - old_x * old_x;
            self.sum_yy += y * y - old_y * old_y;
            self.sum_xy += x * y - old_x * old_y;
        } else {
            self.sum_x += x;
            self.sum_y += y;
            self.sum_xx += x * x;
            self.sum_yy += y * y;
            self.sum_xy += x * y;
        }
    }

    /// Returns the sample covariance:
    /// $$\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$$
    pub fn covariance(&self) -> f64 {
        let n = self.window_x.len();
        if n < 2 {
            return 0.0;
        }
        let n_f = n as f64;
        let mean_x = self.sum_x / n_f;
        let mean_y = self.sum_y / n_f;
        (self.sum_xy - n_f * mean_x * mean_y) / (n_f - 1.0)
    }

    /// Returns the Pearson correlation coefficient $r \in [-1, 1]$:
    /// $$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$
    pub fn correlation(&self) -> f64 {
        let n = self.window_x.len();
        if n < 2 {
            return 0.0;
        }
        let n_f = n as f64;
        let var_x = (self.sum_xx - (self.sum_x * self.sum_x) / n_f) / (n_f - 1.0);
        let var_y = (self.sum_yy - (self.sum_y * self.sum_y) / n_f) / (n_f - 1.0);

        let denom = (var_x * var_y).sqrt();
        if denom <= 1e-12 {
            0.0
        } else {
            (self.covariance() / denom).clamp(-1.0, 1.0)
        }
    }

    /// Computes the OLS Beta ($\beta = \frac{\text{Cov}(X, Y)}{\text{Var}(Y)}$):
    pub fn beta(&self) -> f64 {
        let n = self.window_x.len();
        if n < 2 {
            return 0.0;
        }
        let n_f = n as f64;
        let var_y = (self.sum_yy - (self.sum_y * self.sum_y) / n_f) / (n_f - 1.0);
        if var_y <= 1e-12 {
            0.0
        } else {
            self.covariance() / var_y
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_window_fifo_semantics() {
        let mut rw = RollingWindow::new(3);
        assert!(rw.is_empty());
        assert_eq!(rw.len(), 0);

        assert_eq!(rw.push(10), None);
        assert_eq!(rw.push(20), None);
        assert_eq!(rw.push(30), None);
        assert!(rw.is_full());
        assert_eq!(rw.len(), 3);
        assert_eq!(rw.latest(), Some(&30));
        assert_eq!(rw.oldest(), Some(&10));

        // Pushing 4th item evicts 10
        let evicted = rw.push(40);
        assert_eq!(evicted, Some(10));
        assert_eq!(rw.latest(), Some(&40));
        assert_eq!(rw.oldest(), Some(&20));

        let items: Vec<i32> = rw.iter().cloned().collect();
        assert_eq!(items, vec![20, 30, 40]);
    }

    #[test]
    fn test_online_sliding_stats_numerical_parity() {
        let window_size = 4;
        let mut stats = OnlineSlidingStats::new(window_size);
        let stream = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        for (i, &val) in stream.iter().enumerate() {
            stats.push(val);

            // Compute expected batch metrics on the exact slice
            let start = i.saturating_sub(window_size - 1);
            let slice = &stream[start..=i];
            let n = slice.len() as f64;
            let expected_mean = slice.iter().sum::<f64>() / n;
            let expected_var = if n > 1.0 {
                slice.iter().map(|&x| (x - expected_mean).powi(2)).sum::<f64>() / (n - 1.0)
            } else {
                0.0
            };

            assert!((stats.mean() - expected_mean).abs() < 1e-9, "Mean drift at {i}");
            assert!((stats.variance() - expected_var).abs() < 1e-7, "Var drift at {i}");
        }
    }

    #[test]
    fn test_ema_convergence() {
        let mut ema = ExponentialMovingAverage::with_span(9);
        let mut vals = Vec::new();
        for _ in 0..50 {
            vals.push(ema.update(100.0));
        }
        assert!((vals.last().unwrap() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_sliding_covariance_and_correlation() {
        let mut cov_tracker = OnlineSlidingCovariance::new(5);
        // Perfectly correlated data: y = 2x + 1
        for i in 1..=10 {
            let x = i as f64;
            let y = 2.0 * x + 1.0;
            cov_tracker.push(x, y);
        }

        let r = cov_tracker.correlation();
        assert!((r - 1.0).abs() < 1e-7, "Expected correlation 1.0, got {r}");

        let beta = cov_tracker.beta();
        // Cov(x, y) / Var(y) = Cov(x, 2x) / Var(2x) = 2 Var(x) / (4 Var(x)) = 0.5
        assert!((beta - 0.5).abs() < 1e-7, "Expected beta 0.5, got {beta}");
    }
}
