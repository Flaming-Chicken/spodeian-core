//! CPU and WGSL kernels for Tree Speculative Medusa Eagle GoT and Jacobi.

/// WGSL compute shader for generating 2D causal tree attention mask from node ancestry table.
pub const TREE_ATTENTION_MASK_WGSL: &str = r#"
struct Uniforms {
    num_nodes: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> parent_indices: array<f32>; // [num_nodes]
@group(0) @binding(2) var<storage, read_write> tree_mask: array<f32>; // [num_nodes, num_nodes]

@compute @workgroup_size(16, 16, 1)
fn tree_mask_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; // query node
    let j = gid.y; // key node

    if (i >= u.num_nodes || j >= u.num_nodes) { return; }

    var is_ancestor = false;
    var curr = i;
    for (var step = 0u; step < u.num_nodes; step++) {
        if (curr == j) {
            is_ancestor = true;
            break;
        }
        if (curr == 0u) { break; }
        curr = u32(parent_indices[curr]);
    }

    let out_idx = i * u.num_nodes + j;
    if (is_ancestor) {
        tree_mask[out_idx] = 0.0;
    } else {
        tree_mask[out_idx] = -1e9;
    }
}
"#;

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

/// CPU tree attention mask constructor.
pub fn build_tree_mask_cpu(parent_indices: &[f32], num_nodes: usize) -> Vec<f32> {
    let mut mask = vec![-1e9f32; num_nodes * num_nodes];
    for i in 0..num_nodes {
        let mut curr = i;
        let mut visited = std::collections::HashSet::new();
        while !visited.contains(&curr) {
            mask[i * num_nodes + curr] = 0.0;
            visited.insert(curr);
            if curr == 0 {
                break;
            }
            curr = parent_indices[curr] as usize;
        }
    }
    mask
}

/// CPU Jacobi lookahead iteration step.
pub fn jacobi_step_cpu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let total_tokens = batch * seq_len;
    let mut x_next = vec![0.0f32; total_tokens * dim];
    let mut residuals = vec![0.0f32; batch];

    for t in 0..total_tokens {
        for o in 0..dim {
            let mut dot = bias[o];
            for i in 0..dim {
                dot += x[t * dim + i] * w[i * dim + o];
            }
            x_next[t * dim + o] = dot.tanh();
        }
    }

    for b in 0..batch {
        let mut max_r = 0.0f32;
        for s in 0..seq_len {
            for d in 0..dim {
                let idx = (b * seq_len + s) * dim + d;
                let diff = (x_next[idx] - x[idx]).abs();
                if diff > max_r {
                    max_r = diff;
                }
            }
        }
        residuals[b] = max_r;
    }

    (x_next, residuals)
}
