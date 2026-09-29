//! Dynamic multi-horizon, monotonic non-crossing quantile, and continuous flow heads for Candle.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Dynamic Multi-Horizon Forecast Head.
///
/// Projects an encoder representation $h \in \mathbb{R}^d$ across an arbitrary
/// set of future forecasting horizons (e.g. $[1, 2, 5, 10, 21, 63, 252]$ days).
pub struct DynamicMultiHorizonHead {
    pub horizons: Vec<usize>,
    pub input_dim: usize,
    projection: Linear,
}

impl DynamicMultiHorizonHead {
    /// Create a new multi-horizon head for arbitrary horizon sequences.
    pub fn new(vb: VarBuilder, input_dim: usize, horizons: Vec<usize>) -> CandleResult<Self> {
        let out_dim = horizons.len();
        let projection = linear(input_dim, out_dim, vb.pp("multi_horizon"))?;
        Ok(Self {
            horizons,
            input_dim,
            projection,
        })
    }

    /// Create an initialized mock multi-horizon head with zero weights.
    pub fn new_mock(input_dim: usize, horizons: &[usize], device: &Device) -> CandleResult<Self> {
        let mut tensors = std::collections::HashMap::new();
        let weight = Tensor::zeros((horizons.len(), input_dim), DType::F32, device)?;
        let bias = Tensor::zeros(horizons.len(), DType::F32, device)?;
        tensors.insert("multi_horizon.weight".to_string(), weight);
        tensors.insert("multi_horizon.bias".to_string(), bias);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        Self::new(vb, input_dim, horizons.to_vec())
    }

    /// Forward pass producing predictions for all horizons: shape $(B, K)$ where $K = \text{horizons.len()}$.
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.projection.forward(xs)
    }

    /// Predict scalar forecasts paired with their horizon labels: `[(horizon, forecast)]`.
    pub fn predict_horizons(&self, features: &[f32], device: &Device) -> CandleResult<Vec<(usize, f32)>> {
        let input_tensor = Tensor::from_slice(features, (1, self.input_dim), device)?;
        let output = self.forward(&input_tensor)?;
        let flat = output.flatten_all()?;
        let vals: Vec<f32> = flat.to_vec1()?;

        let result = self
            .horizons
            .iter()
            .copied()
            .zip(vals)
            .collect();
        Ok(result)
    }
}

/// Monotonic Non-Crossing Quantile Head.
///
/// Guarantees strict monotonicity across arbitrary quantile levels $\tau_0 < \tau_1 < \dots < \tau_{M-1}$
/// by predicting cumulative non-negative deltas:
/// $$\hat{q}_0 = \text{fc}_0(h)$$
/// $$\hat{q}_i = \hat{q}_{i-1} + \text{softplus}(\text{fc}_i(h))$$
/// where $\text{softplus}(z) = \ln(1 + e^z)$.
pub struct MonotonicQuantileHead {
    pub quantiles: Vec<f32>,
    pub input_dim: usize,
    base_linear: Linear,
    delta_linears: Vec<Linear>,
}

impl MonotonicQuantileHead {
    /// Initialize a monotonic quantile head for arbitrary quantiles.
    ///
    /// Requires `quantiles` to be sorted strictly ascending in $(0, 1)$.
    pub fn new(vb: VarBuilder, input_dim: usize, mut quantiles: Vec<f32>) -> CandleResult<Self> {
        quantiles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(!quantiles.is_empty(), "Quantiles list must not be empty");

        let base_linear = linear(input_dim, 1, vb.pp("quantile_base"))?;
        let mut delta_linears = Vec::with_capacity(quantiles.len().saturating_sub(1));

        for i in 1..quantiles.len() {
            let delta = linear(input_dim, 1, vb.pp(format!("quantile_delta_{i}")))?;
            delta_linears.push(delta);
        }

        Ok(Self {
            quantiles,
            input_dim,
            base_linear,
            delta_linears,
        })
    }

    /// Create an initialized mock monotonic quantile head with zero weights.
    pub fn new_mock(input_dim: usize, quantiles: &[f32], device: &Device) -> CandleResult<Self> {
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "quantile_base.weight".to_string(),
            Tensor::zeros((1, input_dim), DType::F32, device)?,
        );
        tensors.insert(
            "quantile_base.bias".to_string(),
            Tensor::zeros(1, DType::F32, device)?,
        );
        for i in 1..quantiles.len() {
            tensors.insert(
                format!("quantile_delta_{i}.weight"),
                Tensor::zeros((1, input_dim), DType::F32, device)?,
            );
            tensors.insert(
                format!("quantile_delta_{i}.bias"),
                Tensor::zeros(1, DType::F32, device)?,
            );
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        Self::new(vb, input_dim, quantiles.to_vec())
    }

    /// Forward pass returning monotonic quantile predictions of shape $(B, M)$.
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let base = self.base_linear.forward(xs)?; // (B, 1)
        let mut quantiles = vec![base];

        let mut prev = quantiles[0].clone();
        for delta_layer in &self.delta_linears {
            let raw_delta = delta_layer.forward(xs)?; // (B, 1)
            // softplus(z) = ln(1 + exp(z))
            // In candle: (1 + z.exp()).log()
            let softplus_delta = (raw_delta.exp()? + 1.0)?.log()?;
            let current = (&prev + &softplus_delta)?;
            quantiles.push(current.clone());
            prev = current;
        }

        Tensor::cat(&quantiles, 1)
    }

    /// Predict monotonic quantiles for a feature slice on CPU: returns `[(tau, q_val)]`.
    pub fn predict_quantiles(&self, features: &[f32], device: &Device) -> CandleResult<Vec<(f32, f32)>> {
        let input_tensor = Tensor::from_slice(features, (1, self.input_dim), device)?;
        let output = self.forward(&input_tensor)?;
        let flat = output.flatten_all()?;
        let vals: Vec<f32> = flat.to_vec1()?;

        let result = self
            .quantiles
            .iter()
            .copied()
            .zip(vals)
            .collect();
        Ok(result)
    }

    /// Compute pinball loss for quantile $\tau$:
    /// $$\mathcal{L}_\tau(y, \hat{q}) = \max(\tau(y - \hat{q}), (\tau - 1)(y - \hat{q}))$$
    pub fn pinball_loss(tau: f32, y: f32, q_hat: f32) -> f32 {
        let error = y - q_hat;
        if error > 0.0 {
            tau * error
        } else {
            (tau - 1.0) * error
        }
    }
}

/// Continuous Flow Velocity Head for RK4 Flow Matching.
///
/// Projects spatial states $x \in \mathbb{R}^d$ and continuous flow time $t \in [0, 1]$
/// into continuous velocity vector field $v_\theta(x, t) \in \mathbb{R}^d$.
pub struct VelocityFlowHead {
    pub dim: usize,
    pub num_freqs: usize,
    time_proj: Linear,
    hidden_linear: Linear,
    out_linear: Linear,
}

impl VelocityFlowHead {
    pub fn new(vb: VarBuilder, dim: usize, num_freqs: usize, hidden_dim: usize) -> CandleResult<Self> {
        let time_dim = num_freqs * 2;
        let time_proj = linear(time_dim, hidden_dim, vb.pp("flow_time_proj"))?;
        let hidden_linear = linear(dim + hidden_dim, hidden_dim, vb.pp("flow_hidden"))?;
        let out_linear = linear(hidden_dim, dim, vb.pp("flow_out"))?;

        Ok(Self {
            dim,
            num_freqs,
            time_proj,
            hidden_linear,
            out_linear,
        })
    }

    /// Generate sinusoidal positional embedding for scalar flow time $t \in [0, 1]$.
    pub fn time_embedding(&self, t: f32) -> Vec<f32> {
        let mut emb = Vec::with_capacity(self.num_freqs * 2);
        for i in 0..self.num_freqs {
            let freq = 2.0f32.powi(i as i32) * std::f32::consts::PI;
            emb.push((freq * t).sin());
            emb.push((freq * t).cos());
        }
        emb
    }

    /// Evaluate velocity vector $v_\theta(x, t) \in \mathbb{R}^d$ given state $x$ and time $t$.
    pub fn predict_velocity(&self, x: &[f32], t: f32, device: &Device) -> CandleResult<Vec<f32>> {
        assert_eq!(x.len(), self.dim);
        let t_emb = self.time_embedding(t);
        let t_tensor = Tensor::from_slice(&t_emb, (1, t_emb.len()), device)?;
        let x_tensor = Tensor::from_slice(x, (1, self.dim), device)?;

        let t_hidden = self.time_proj.forward(&t_tensor)?.gelu()?;
        let combined = Tensor::cat(&[&x_tensor, &t_hidden], 1)?;
        let h = self.hidden_linear.forward(&combined)?.gelu()?;
        let v = self.out_linear.forward(&h)?;

        let flat = v.flatten_all()?;
        flat.to_vec1::<f32>()
    }
}

/// Continuous-Time Horizon Forecast Head.
///
/// Predicts expected returns or signals for ANY continuous future horizon $t \in \mathbb{R}^+$
/// (e.g. $t = 0.5$ intraday, $t = 2.5$ days, $t = 21.0$ days, $t = 252.0$ days) using continuous
/// logarithmic-Fourier temporal basis projections:
/// $$\Phi_k(t) = [\sin(\omega_k \ln(1 + t)), \cos(\omega_k \ln(1 + t))]$$
pub struct ContinuousHorizonHead {
    pub input_dim: usize,
    pub num_freqs: usize,
    pub hidden_dim: usize,
    feat_proj: Linear,
    time_proj: Linear,
    out_linear: Linear,
}

impl ContinuousHorizonHead {
    pub fn new(vb: VarBuilder, input_dim: usize, num_freqs: usize, hidden_dim: usize) -> CandleResult<Self> {
        let time_dim = num_freqs * 2;
        let feat_proj = linear(input_dim, hidden_dim, vb.pp("horizon_feat_proj"))?;
        let time_proj = linear(time_dim, hidden_dim, vb.pp("horizon_time_proj"))?;
        let out_linear = linear(hidden_dim, 1, vb.pp("horizon_out"))?;

        Ok(Self {
            input_dim,
            num_freqs,
            hidden_dim,
            feat_proj,
            time_proj,
            out_linear,
        })
    }

    /// Creates an initialized mock continuous horizon head with zero weights.
    pub fn new_mock(input_dim: usize, num_freqs: usize, hidden_dim: usize, device: &Device) -> CandleResult<Self> {
        let time_dim = num_freqs * 2;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "horizon_feat_proj.weight".to_string(),
            Tensor::zeros((hidden_dim, input_dim), DType::F32, device)?,
        );
        tensors.insert(
            "horizon_feat_proj.bias".to_string(),
            Tensor::zeros(hidden_dim, DType::F32, device)?,
        );
        tensors.insert(
            "horizon_time_proj.weight".to_string(),
            Tensor::zeros((hidden_dim, time_dim), DType::F32, device)?,
        );
        tensors.insert(
            "horizon_time_proj.bias".to_string(),
            Tensor::zeros(hidden_dim, DType::F32, device)?,
        );
        tensors.insert(
            "horizon_out.weight".to_string(),
            Tensor::zeros((1, hidden_dim), DType::F32, device)?,
        );
        tensors.insert(
            "horizon_out.bias".to_string(),
            Tensor::zeros(1, DType::F32, device)?,
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        Self::new(vb, input_dim, num_freqs, hidden_dim)
    }

    /// Continuous logarithmic-Fourier temporal basis embedding for horizon $t \ge 0$.
    pub fn time_embedding(&self, t: f32) -> Vec<f32> {
        let log_t = (1.0 + t.max(0.0)).ln();
        let mut emb = Vec::with_capacity(self.num_freqs * 2);
        for i in 0..self.num_freqs {
            let freq = 1.0 / (10.0f32.powf(i as f32 / self.num_freqs.max(1) as f32));
            emb.push((freq * log_t).sin());
            emb.push((freq * log_t).cos());
        }
        emb
    }

    /// Predict expected alpha for an arbitrary continuous horizon $t \in \mathbb{R}^+$.
    pub fn predict_horizon(&self, features: &[f32], t: f32, device: &Device) -> CandleResult<f32> {
        assert_eq!(features.len(), self.input_dim);
        let t_emb = self.time_embedding(t);
        let t_tensor = Tensor::from_slice(&t_emb, (1, t_emb.len()), device)?;
        let feat_tensor = Tensor::from_slice(features, (1, self.input_dim), device)?;

        let feat_h = self.feat_proj.forward(&feat_tensor)?;
        let time_h = self.time_proj.forward(&t_tensor)?;
        let h = (feat_h + time_h)?.gelu()?;
        let out = self.out_linear.forward(&h)?;

        let flat = out.flatten_all()?;
        let vals: Vec<f32> = flat.to_vec1()?;
        Ok(vals[0])
    }

    /// Predict expected alpha across multiple arbitrary continuous horizons: `[(t, alpha(t))]`.
    pub fn predict_continuous_horizons(
        &self,
        features: &[f32],
        horizons: &[f32],
        device: &Device,
    ) -> CandleResult<Vec<(f32, f32)>> {
        let mut results = Vec::with_capacity(horizons.len());
        for &t in horizons {
            let pred = self.predict_horizon(features, t, device)?;
            results.push((t, pred));
        }
        Ok(results)
    }
}

