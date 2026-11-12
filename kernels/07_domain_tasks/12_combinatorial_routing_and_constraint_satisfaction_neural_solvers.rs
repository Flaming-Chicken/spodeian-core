//! CPU and WGSL kernels for Combinatorial Routing and Constraint Satisfaction Neural Solvers.

/// CPU linear anchor: `y[b, o] = bias[o] + sum_i x[b, i] * w[i, o]`.
pub fn linear_anchor_cpu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_out];
    for (b, row) in y.chunks_mut(d_out).enumerate() {
        for (o, out) in row.iter_mut().enumerate() {
            *out = bias[o]
                + (0..d_in)
                    .map(|i| x[b * d_in + i] * w[i * d_out + o])
                    .sum::<f32>();
        }
    }
    y
}

/// Fused TSP Closed-Loop Tour Length computation kernel:
/// coords: [B, N, 2] row-major -> index = (bi * N + ni) * 2 + coord_dim
/// tour: [B, N] row-major -> index = bi * N + ni
/// out: [B]
pub fn tsp_tour_cost_cpu(coords: &[f32], tour: &[usize], b: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b];
    for bi in 0..b {
        let mut total_len = 0.0f32;
        let b_tour_offset = bi * n;
        let b_coord_offset = bi * n * 2;

        for i in 0..n {
            let curr_node = tour[b_tour_offset + i];
            let next_node = tour[b_tour_offset + ((i + 1) % n)];

            let x1 = coords[b_coord_offset + curr_node * 2];
            let y1 = coords[b_coord_offset + curr_node * 2 + 1];

            let x2 = coords[b_coord_offset + next_node * 2];
            let y2 = coords[b_coord_offset + next_node * 2 + 1];

            let dx = x1 - x2;
            let dy = y1 - y2;
            total_len += (dx * dx + dy * dy + 1e-12f32).sqrt();
        }
        out[bi] = total_len;
    }
    out
}

/// Fused TSP Tour Cost WGSL compute shader
pub const TSP_TOUR_COST_WGSL: &str = r#"
struct Config {
    b: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> coords: array<f32>;
@group(0) @binding(1) var<storage, read> tour: array<u32>;
@group(0) @binding(2) var<storage, read_write> costs: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bi = global_id.x;
    if (bi >= cfg.b) {
        return;
    }

    var total_len: f32 = 0.0;
    let b_tour_offset = bi * cfg.n;
    let b_coord_offset = bi * cfg.n * 2u;

    for (var i: u32 = 0u; i < cfg.n; i = i + 1u) {
        let curr_node = tour[b_tour_offset + i];
        let next_idx = (i + 1u) % cfg.n;
        let next_node = tour[b_tour_offset + next_idx];

        let x1 = coords[b_coord_offset + curr_node * 2u];
        let y1 = coords[b_coord_offset + curr_node * 2u + 1u];

        let x2 = coords[b_coord_offset + next_node * 2u];
        let y2 = coords[b_coord_offset + next_node * 2u + 1u];

        let dx = x1 - x2;
        let dy = y1 - y2;
        total_len = total_len + sqrt(dx * dx + dy * dy + 1e-12);
    }

    costs[bi] = total_len;
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn tsp_tour_cost_auto(coords: &[f32], tour: &[usize], b: usize, n: usize) -> Vec<f32> {
    tsp_tour_cost_cpu(coords, tour, b, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsp_tour_cost_cpu() {
        // 4 corners of unit square: (0,0), (1,0), (1,1), (0,1)
        let coords = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let tour = vec![0, 1, 2, 3];
        let costs = tsp_tour_cost_cpu(&coords, &tour, 1, 4);
        assert_eq!(costs.len(), 1);
        // Perimeter of unit square is 4.0
        assert!((costs[0] - 4.0).abs() < 1e-4);
    }
}
