//! CPU (and optional WGSL) kernels for Energy-Based Models, RBM, and Hopfield Networks.

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

/// Overdamped Langevin dynamics MCMC sampling step on CPU: x_next = x - 0.5*eps*grad_E + sqrt(eps)*noise.
pub fn langevin_dynamics_step_cpu(
    x: &[f32],
    grad_energy: &[f32],
    noise: &[f32],
    step_size: f32,
) -> Vec<f32> {
    let sqrt_step = step_size.sqrt();
    let half_step = 0.5 * step_size;
    x.iter()
        .zip(grad_energy.iter())
        .zip(noise.iter())
        .map(|((&xi, &gi), &ni)| xi - half_step * gi + sqrt_step * ni)
        .collect()
}

/// RBM free energy calculation on CPU.
pub fn rbm_free_energy_cpu(
    v: &[f32],
    w: &[f32],
    a: &[f32],
    b: &[f32],
    batch: usize,
    v_dim: usize,
    h_dim: usize,
) -> Vec<f32> {
    let mut energies = Vec::with_capacity(batch);

    for i in 0..batch {
        let v_row = &v[i * v_dim..(i + 1) * v_dim];
        let mut v_bias = 0.0f32;
        for d in 0..v_dim {
            v_bias += v_row[d] * a[d];
        }

        let mut h_term = 0.0f32;
        for j in 0..h_dim {
            let mut pre_act = b[j];
            for d in 0..v_dim {
                pre_act += v_row[d] * w[d * h_dim + j];
            }
            h_term += (pre_act.exp() + 1.0).ln();
        }

        energies.push(-v_bias - h_term);
    }
    energies
}

/// WGSL compute shader for massively parallel Langevin dynamics updates.
pub const WGSL_LANGEVIN_STEP: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> grad_energy: array<f32>;
@group(0) @binding(2) var<storage, read> noise: array<f32>;
@group(0) @binding(3) var<storage, read_write> x_next: array<f32>;

struct Uniforms {
    total_elements: u32,
    step_size: f32,
    sqrt_step: f32,
};
@group(0) @binding(4) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.total_elements) {
        return;
    }
    let xi = x[idx];
    let gi = grad_energy[idx];
    let ni = noise[idx];

    x_next[idx] = xi - 0.5 * u.step_size * gi + u.sqrt_step * ni;
}
"#;
