/// WGSL compute shader for fused Block-Diagonal OFT forward pass: y = (x @ R) @ W0.
pub const FUSED_BLOCK_OFT_FORWARD_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    d_in: u32,
    d_out: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>; // [batch, d_in]
@group(0) @binding(2) var<storage, read> r_matrix: array<f32>; // [d_in, d_in]
@group(0) @binding(3) var<storage, read> w0: array<f32>; // [d_in, d_out]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [batch, d_out]

@compute @workgroup_size(16, 16, 1)
fn oft_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if (row >= u.batch || col >= u.d_out) { return; }

    var acc: f32 = 0.0;
    for (var m = 0u; m < u.d_in; m++) {
        var x_rot_m: f32 = 0.0;
        for (var k = 0u; k < u.d_in; k++) {
            x_rot_m += x[row * u.d_in + k] * r_matrix[k * u.d_in + m];
        }
        acc += x_rot_m * w0[m * u.d_out + col];
    }

    y[row * u.d_out + col] = acc;
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

/// CPU Block-Diagonal OFT forward pass: `y = (x @ R) @ W0`.
pub fn block_oft_forward_cpu(
    x: &[f32],
    r_matrix: &[f32],
    w0: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_out];

    for row in 0..batch {
        let mut x_rot = vec![0.0f32; d_in];
        for m in 0..d_in {
            let mut dot = 0.0f32;
            for k in 0..d_in {
                dot += x[row * d_in + k] * r_matrix[k * d_in + m];
            }
            x_rot[m] = dot;
        }

        for col in 0..d_out {
            let mut acc = 0.0f32;
            for m in 0..d_in {
                acc += x_rot[m] * w0[m * d_out + col];
            }
            y[row * d_out + col] = acc;
        }
    }

    y
}
