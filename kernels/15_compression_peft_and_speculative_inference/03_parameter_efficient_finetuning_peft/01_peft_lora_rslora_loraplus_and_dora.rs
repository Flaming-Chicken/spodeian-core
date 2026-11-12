//! CPU and WGSL kernels for PEFT LoRA Rslora Loraplus and DoRA.

/// WGSL compute shader for fused low-rank forward pass: y = x W0 + scale * (x A) B.
pub const FUSED_LORA_FORWARD_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    d_in: u32,
    d_out: u32,
    rank: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>; // [batch, d_in]
@group(0) @binding(2) var<storage, read> w0: array<f32>; // [d_in, d_out]
@group(0) @binding(3) var<storage, read> a: array<f32>; // [d_in, rank]
@group(0) @binding(4) var<storage, read> b: array<f32>; // [rank, d_out]
@group(0) @binding(5) var<storage, read_write> y: array<f32>; // [batch, d_out]

@compute @workgroup_size(16, 16, 1)
fn lora_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x; // batch index
    let col = gid.y; // output feature index

    if (row >= u.batch || col >= u.d_out) { return; }

    // 1. Base linear component: x @ W0
    var base_val: f32 = 0.0;
    for (var i = 0u; i < u.d_in; i++) {
        base_val += x[row * u.d_in + i] * w0[i * u.d_out + col];
    }

    // 2. Low-rank adapter component: (x @ A) @ B
    var adapter_val: f32 = 0.0;
    for (var r = 0u; r < u.rank; r++) {
        var x_a: f32 = 0.0;
        for (var i = 0u; i < u.d_in; i++) {
            x_a += x[row * u.d_in + i] * a[i * u.rank + r];
        }
        adapter_val += x_a * b[r * u.d_out + col];
    }

    y[row * u.d_out + col] = base_val + u.scale * adapter_val;
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

/// CPU implementation of fused LoRA forward pass.
pub fn lora_forward_cpu(
    x: &[f32],
    w0: &[f32],
    a: &[f32],
    b: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
    rank: usize,
    scale: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_out];

    for row in 0..batch {
        // Compute x @ A -> [rank]
        let mut x_a = vec![0.0f32; rank];
        for r in 0..rank {
            for i in 0..d_in {
                x_a[r] += x[row * d_in + i] * a[i * rank + r];
            }
        }

        for col in 0..d_out {
            let mut base_val = 0.0f32;
            for i in 0..d_in {
                base_val += x[row * d_in + i] * w0[i * d_out + col];
            }

            let mut adapter_val = 0.0f32;
            for r in 0..rank {
                adapter_val += x_a[r] * b[r * d_out + col];
            }

            y[row * d_out + col] = base_val + scale * adapter_val;
        }
    }

    y
}
