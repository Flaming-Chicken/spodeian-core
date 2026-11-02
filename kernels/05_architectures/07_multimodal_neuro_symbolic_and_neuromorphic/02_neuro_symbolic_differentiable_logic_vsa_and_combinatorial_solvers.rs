//! CPU (and optional WGSL) kernels for Neuro Symbolic Differentiable Logic Vsa and Combinatorial Solvers.

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

/// CPU kernel for Łukasiewicz fuzzy logic operators: returns (t_norm, t_conorm, implication).
pub fn lukasiewicz_logic_cpu(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(a.len(), b.len());
    let mut t_norm = Vec::with_capacity(a.len());
    let mut t_conorm = Vec::with_capacity(a.len());
    let mut implication = Vec::with_capacity(a.len());

    for i in 0..a.len() {
        let ai = a[i];
        let bi = b[i];
        t_norm.push((ai + bi - 1.0).clamp(0.0, 1.0));
        t_conorm.push((ai + bi).clamp(0.0, 1.0));
        implication.push((1.0 - ai + bi).clamp(0.0, 1.0));
    }
    (t_norm, t_conorm, implication)
}

/// WGSL shader for fused Łukasiewicz t-norm, t-conorm, and implication.
pub const LOGIC_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> t_norm: array<f32>;
@group(0) @binding(3) var<storage, read_write> t_conorm: array<f32>;
@group(0) @binding(4) var<storage, read_write> implication: array<f32>;

struct Uniforms {
    total_elements: u32,
};
@group(0) @binding(5) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= u.total_elements) {
        return;
    }
    let ai = a[idx];
    let bi = b[idx];
    t_norm[idx] = clamp(ai + bi - 1.0, 0.0, 1.0);
    t_conorm[idx] = clamp(ai + bi, 0.0, 1.0);
    implication[idx] = clamp(1.0 - ai + bi, 0.0, 1.0);
}
"#;
