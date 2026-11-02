//! CPU (and optional WGSL) kernels for Memory Optimization Selective Checkpointing Offloading and Grad Hooks.

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

/// WGSL compute shader for selective recomputation of fused SiLU gate backward:
/// Recomputes a = z * sigmoid(z) on-the-fly and evaluates gz = (grad_out * gate) * (sigmoid(z) * (1 + z * (1 - sigmoid(z))))
pub const SELECTIVE_RECOMPUTE_SILU_GATE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> z_in: array<f32>;
@group(0) @binding(1) var<storage, read> gate_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> gz_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_gate_out: array<f32>;

struct Params {
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn recompute_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let z = z_in[idx];
    let g = gate_in[idx];
    let g_out = grad_out_in[idx];

    let sig_z = 1.0 / (1.0 + exp(-z));
    let a = z * sig_z;

    let da_dz = sig_z * (1.0 + z * (1.0 - sig_z));
    let g_a = g_out * g;

    gz_out[idx] = g_a * da_dz;
    grad_gate_out[idx] = g_out * a;
}
"#;

/// CPU reference implementation for selective recompute SiLU gate backward.
pub fn selective_recompute_silu_gate_cpu(
    z: &[f32],
    gate: &[f32],
    grad_out: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut gz_out = Vec::with_capacity(z.len());
    let mut grad_gate_out = Vec::with_capacity(z.len());

    for i in 0..z.len() {
        let zv = z[i];
        let gv = gate[i];
        let gov = grad_out[i];

        let sig_z = 1.0f32 / (1.0f32 + (-zv).exp());
        let a = zv * sig_z;
        let da_dz = sig_z * (1.0f32 + zv * (1.0f32 - sig_z));
        let g_a = gov * gv;

        gz_out.push(g_a * da_dz);
        grad_gate_out.push(gov * a);
    }

    (gz_out, grad_gate_out)
}
