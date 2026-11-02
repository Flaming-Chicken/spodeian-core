//! CPU and WGSL kernels for Knowledge Distillation Logit Feature and Minillm.

/// WGSL compute shader for fused temperature scaling, softmax, and KL divergence reduction.
pub const FUSED_SOFTMAX_KL_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    vocab_size: u32,
    temperature: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> student_logits: array<f32>; // [batch, vocab_size]
@group(0) @binding(2) var<storage, read> teacher_logits: array<f32>; // [batch, vocab_size]
@group(0) @binding(3) var<storage, read_write> kl_per_seq: array<f32>; // [batch]

@compute @workgroup_size(64, 1, 1)
fn fused_kl_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= u.batch) { return; }

    let t_inv = 1.0 / u.temperature;
    var max_s: f32 = -1e9;
    var max_t: f32 = -1e9;

    for (var i = 0u; i < u.vocab_size; i++) {
        let s_val = student_logits[b * u.vocab_size + i] * t_inv;
        let t_val = teacher_logits[b * u.vocab_size + i] * t_inv;
        if (s_val > max_s) { max_s = s_val; }
        if (t_val > max_t) { max_t = t_val; }
    }

    var sum_exp_s: f32 = 0.0;
    var sum_exp_t: f32 = 0.0;
    for (var i = 0u; i < u.vocab_size; i++) {
        sum_exp_s += exp(student_logits[b * u.vocab_size + i] * t_inv - max_s);
        sum_exp_t += exp(teacher_logits[b * u.vocab_size + i] * t_inv - max_t);
    }
    let log_z_s = max_s + log(sum_exp_s);
    let log_z_t = max_t + log(sum_exp_t);

    var kl_sum: f32 = 0.0;
    for (var i = 0u; i < u.vocab_size; i++) {
        let log_p_s = student_logits[b * u.vocab_size + i] * t_inv - log_z_s;
        let log_p_t = teacher_logits[b * u.vocab_size + i] * t_inv - log_z_t;
        let p_t = exp(log_p_t);
        kl_sum += p_t * (log_p_t - log_p_s);
    }

    kl_per_seq[b] = kl_sum;
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

/// CPU forward KL divergence D_KL(P_T || P_S).
pub fn kl_divergence_cpu(
    student_logits: &[f32],
    teacher_logits: &[f32],
    batch: usize,
    vocab: usize,
    temp: f32,
) -> f32 {
    let mut total_kl = 0.0f32;
    for b in 0..batch {
        let s_row = &student_logits[b * vocab..(b + 1) * vocab];
        let t_row = &teacher_logits[b * vocab..(b + 1) * vocab];

        let max_s = s_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) / temp;
        let max_t = t_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) / temp;

        let sum_exp_s: f32 = s_row.iter().map(|&x| ((x / temp) - max_s).exp()).sum();
        let sum_exp_t: f32 = t_row.iter().map(|&x| ((x / temp) - max_t).exp()).sum();

        let log_z_s = max_s + sum_exp_s.ln();
        let log_z_t = max_t + sum_exp_t.ln();

        let mut kl = 0.0f32;
        for i in 0..vocab {
            let log_p_s = (s_row[i] / temp) - log_z_s;
            let log_p_t = (t_row[i] / temp) - log_z_t;
            let p_t = log_p_t.exp();
            kl += p_t * (log_p_t - log_p_s);
        }
        total_kl += kl;
    }
    total_kl / batch as f32
}

/// CPU reverse KL divergence D_KL(P_S || P_T).
pub fn reverse_kl_cpu(
    student_logits: &[f32],
    teacher_logits: &[f32],
    batch: usize,
    vocab: usize,
    temp: f32,
) -> f32 {
    let mut total_rev = 0.0f32;
    for b in 0..batch {
        let s_row = &student_logits[b * vocab..(b + 1) * vocab];
        let t_row = &teacher_logits[b * vocab..(b + 1) * vocab];

        let max_s = s_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) / temp;
        let max_t = t_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) / temp;

        let sum_exp_s: f32 = s_row.iter().map(|&x| ((x / temp) - max_s).exp()).sum();
        let sum_exp_t: f32 = t_row.iter().map(|&x| ((x / temp) - max_t).exp()).sum();

        let log_z_s = max_s + sum_exp_s.ln();
        let log_z_t = max_t + sum_exp_t.ln();

        let mut kl = 0.0f32;
        for i in 0..vocab {
            let log_p_s = (s_row[i] / temp) - log_z_s;
            let log_p_t = (t_row[i] / temp) - log_z_t;
            let p_s = log_p_s.exp();
            kl += p_s * (log_p_s - log_p_t);
        }
        total_rev += kl;
    }
    total_rev / batch as f32
}
