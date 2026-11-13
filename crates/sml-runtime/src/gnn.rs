//! Temporal Graph Networks (TGN) & Dynamic Graph Operators (`sml-runtime::gnn`).
//!
//! Grounding in relevant literature:
//! 1. **Temporal Graph Networks for Deep Learning on Dynamic Graphs (Rossi et al., 2020)**:
//!    Continuous-time event stream processing with memory modules updated via continuous
//!    Fourier time embeddings and recurrent updates (GRU/LSTM).
//! 2. **Continuous-Time Dynamic Network Embeddings (Kazemi et al., 2019)**:
//!    Harmonic Fourier time delta representations $\Phi_k(\Delta t) = \cos(\omega_k \Delta t)$.

use serde::{Deserialize, Serialize};

/// A timestamped interaction or shock event across two graph nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInteractionEvent {
    pub source_id: usize,
    pub target_id: usize,
    /// Continuous timestamp (e.g. Unix seconds or fractional days)
    pub timestamp: f64,
    /// Shock magnitude or transmitted factor loading
    pub shock_magnitude: f64,
    pub linkage_type: String,
}

/// Time-aware continuous memory state for an asset or entity node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TgnNodeMemory {
    pub node_id: usize,
    pub symbol: String,
    pub memory_dim: usize,
    /// Dynamic hidden state representation $\mathbf{s}_i(t) \in \mathbb{R}^d$
    pub state: Vec<f64>,
    /// Last update timestamp $t_{\text{last}}$
    pub last_timestamp: f64,
}

impl TgnNodeMemory {
    pub fn new(node_id: usize, symbol: impl Into<String>, memory_dim: usize) -> Self {
        Self {
            node_id,
            symbol: symbol.into(),
            memory_dim,
            state: vec![0.0; memory_dim],
            last_timestamp: 0.0,
        }
    }

    /// Resets the memory to initial zero state.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.last_timestamp = 0.0;
    }
}

/// Temporal Graph Network (TGN) for tracking dynamic continuous-time interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSpilloverGraph {
    pub memory_dim: usize,
    pub nodes: Vec<TgnNodeMemory>,
    /// Event history buffer for replay and temporal backpropagation
    pub event_log: Vec<TemporalInteractionEvent>,
    /// GRU recurrent transition weights for memory updates
    pub gru_w_z: Vec<f64>,
    pub gru_w_r: Vec<f64>,
    pub gru_w_h: Vec<f64>,
}

impl TemporalSpilloverGraph {
    pub fn new(memory_dim: usize) -> Self {
        assert!(memory_dim > 0, "Memory dimension must be positive");
        let n = memory_dim;
        let mut gru_w_z = vec![0.0; n * n];
        let mut gru_w_r = vec![0.0; n * n];
        let mut gru_w_h = vec![0.0; n * n];

        for i in 0..n {
            gru_w_z[i * n + i] = 0.5;
            gru_w_r[i * n + i] = 0.5;
            gru_w_h[i * n + i] = 0.7;
        }

        Self {
            memory_dim,
            nodes: Vec::new(),
            event_log: Vec::new(),
            gru_w_z,
            gru_w_r,
            gru_w_h,
        }
    }

    /// Register a new node in the temporal graph.
    pub fn add_node(&mut self, symbol: impl Into<String>) -> usize {
        let id = self.nodes.len();
        self.nodes
            .push(TgnNodeMemory::new(id, symbol, self.memory_dim));
        id
    }

    /// Continuous Fourier time-embedding mapping time delta $\Delta t$ into $\mathbb{R}^d$:
    /// $$\Phi_k(\Delta t) = \cos(\omega_k \Delta t)$$
    pub fn compute_time_embedding(&self, dt: f64) -> Vec<f64> {
        let mut embed = Vec::with_capacity(self.memory_dim);
        for k in 0..self.memory_dim {
            let omega = 1.0 / 10.0f64.powf(k as f64 / self.memory_dim as f64);
            embed.push((omega * dt.max(0.0)).cos());
        }
        embed
    }

    /// Ingests a new temporal interaction event and updates the target node's memory via GRU.
    pub fn process_event(&mut self, event: TemporalInteractionEvent) -> Result<(), String> {
        let n_nodes = self.nodes.len();
        if event.source_id >= n_nodes || event.target_id >= n_nodes {
            return Err(format!(
                "Node ID out of bounds: source={}, target={}, total_nodes={}",
                event.source_id, event.target_id, n_nodes
            ));
        }

        let dt = (event.timestamp - self.nodes[event.target_id].last_timestamp).max(0.0);
        let time_embed = self.compute_time_embedding(dt);

        // Compute message vector: m_j = s_source + shock * time_embed
        let d = self.memory_dim;
        let mut message = vec![0.0; d];
        for k in 0..d {
            let s_src = self.nodes[event.source_id].state[k];
            message[k] = s_src + event.shock_magnitude * time_embed[k];
        }

        // Apply GRU update on target node's memory: s_new = GRU(message, s_prev)
        let s_prev = &self.nodes[event.target_id].state;
        let mut s_new = vec![0.0; d];

        for i in 0..d {
            let mut sum_z = 0.0;
            let mut sum_r = 0.0;
            for j in 0..d {
                sum_z += self.gru_w_z[i * d + j] * message[j];
                sum_r += self.gru_w_r[i * d + j] * s_prev[j];
            }
            let z = sigmoid(sum_z);
            let r = sigmoid(sum_r);

            let mut sum_h = 0.0;
            for j in 0..d {
                sum_h += self.gru_w_h[i * d + j] * (r * s_prev[j] + message[j]);
            }
            let h_tilde = sum_h.tanh();

            s_new[i] = (1.0 - z) * s_prev[i] + z * h_tilde;
        }

        self.nodes[event.target_id].state = s_new;
        self.nodes[event.target_id].last_timestamp = event.timestamp;
        self.event_log.push(event);

        Ok(())
    }

    /// Query the current dynamic contagion / influence score of an entity node.
    pub fn get_contagion_score(&self, node_id: usize) -> Result<f64, String> {
        if node_id >= self.nodes.len() {
            return Err(format!("Node index {node_id} out of bounds"));
        }
        let state = &self.nodes[node_id].state;
        let norm_sq: f64 = state.iter().map(|&x| x * x).sum();
        Ok(norm_sq.sqrt())
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-15.0, 15.0)).exp())
}
