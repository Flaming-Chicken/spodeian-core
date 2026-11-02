//! CPU and WGSL kernels for QLoRA, LoftQ, and QA-LoRA Quantized PEFT.

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

/// WGSL compute shader for fused QLoRA forward pass:
/// `Y[b, o] = bias[o] + sum_i (X[b, i] * W[i, o]) + scaling * sum_r ( (sum_i X[b, i] * A[i, r]) * B[r, o] )`
pub const QLORA_FUSED_FORWARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_deq: array<f32>;
@group(0) @binding(2) var<storage, read> lora_a: array<f32>;
@group(0) @binding(3) var<storage, read> lora_b: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_y: array<f32>;

struct Uniforms {
    batch: u32,
    d_in: u32,
    d_out: u32,
    rank: u32,
    scaling: f32,
};
@group(0) @binding(6) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let o = global_id.y;
    if (b >= params.batch || o >= params.d_out) {
        return;
    }

    // 1. Base frozen GEMM
    var base_acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.d_in; i = i + 1u) {
        base_acc = base_acc + x[b * params.d_in + i] * w_deq[i * params.d_out + o];
    }

    // 2. Low-rank adapter GEMM: (X @ A) @ B
    var lora_acc: f32 = 0.0;
    for (var r: u32 = 0u; r < params.rank; r = r + 1u) {
        var x_a: f32 = 0.0;
        for (var i: u32 = 0u; i < params.d_in; i = i + 1u) {
            x_a = x_a + x[b * params.d_in + i] * lora_a[i * params.rank + r];
        }
        lora_acc = lora_acc + x_a * lora_b[r * params.d_out + o];
    }

    let idx = b * params.d_out + o;
    out_y[idx] = bias[o] + base_acc + lora_acc * params.scaling;
}
"#;

/// CPU reference implementation of fused QLoRA base + adapter forward evaluation.
pub fn qlora_forward_cpu(
    x: &[f32],
    w_deq: &[f32],
    lora_a: &[f32],
    lora_b: &[f32],
    bias: &[f32],
    scaling: f32,
    batch: usize,
    d_in: usize,
    d_out: usize,
    rank: usize,
) -> Vec<f32> {
    let mut out_y = vec![0.0f32; batch * d_out];
    for b in 0..batch {
        for o in 0..d_out {
            let mut base_acc = 0.0f32;
            for i in 0..d_in {
                base_acc += x[b * d_in + i] * w_deq[i * d_out + o];
            }

            let mut lora_acc = 0.0f32;
            for r in 0..rank {
                let mut x_a = 0.0f32;
                for i in 0..d_in {
                    x_a += x[b * d_in + i] * lora_a[i * rank + r];
                }
                lora_acc += x_a * lora_b[r * d_out + o];
            }

            out_y[b * d_out + o] = bias[o] + base_acc + lora_acc * scaling;
        }
    }
    out_y
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn qlora_forward_auto(
    x: &[f32],
    w_deq: &[f32],
    lora_a: &[f32],
    lora_b: &[f32],
    bias: &[f32],
    scaling: f32,
    batch: usize,
    d_in: usize,
    d_out: usize,
    rank: usize,
) -> Vec<f32> {
    qlora_forward_cpu(
        x, w_deq, lora_a, lora_b, bias, scaling, batch, d_in, d_out, rank,
    )
}
