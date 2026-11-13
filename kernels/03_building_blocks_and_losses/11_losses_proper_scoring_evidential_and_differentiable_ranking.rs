//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 11:
//! Losses: Proper Scoring Rules, Evidential Deep Learning, and Differentiable Ranking.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// WGSL compute shader for parallel soft ranking computation.
/// For each sequence element (b, i), computes:
/// R_{b, i} = 0.5 + sum_{j=0}^{N-1} sigmoid((scores[b, i] - scores[b, j]) / tau)
pub const SOFT_RANKS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_ranks: array<f32>;

struct Uniforms {
    batch_size: u32,
    n_items: u32,
    tau: f32,
    _pad: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let i = global_id.y;
    if (b >= params.batch_size || i >= params.n_items) {
        return;
    }
    let s_i = scores[b * params.n_items + i];
    var rank_accum: f32 = 0.5;
    for (var j: u32 = 0u; j < params.n_items; j = j + 1u) {
        let s_j = scores[b * params.n_items + j];
        let diff = (s_i - s_j) / params.tau;
        let sig = 1.0 / (1.0 + exp(-diff));
        rank_accum = rank_accum + sig;
    }
    out_ranks[b * params.n_items + i] = rank_accum;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SoftRanksUniforms {
    batch_size: u32,
    n_items: u32,
    tau: f32,
    _pad: u32,
}

/// Linear anchor forward on CPU: Y = X @ W + b.
pub fn linear_anchor_forward_cpu(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch_size: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * out_dim];
    for b in 0..batch_size {
        for j in 0..out_dim {
            let mut sum = bias[j];
            for i in 0..in_dim {
                sum += x[b * in_dim + i] * weight[i * out_dim + j];
            }
            out[b * out_dim + j] = sum;
        }
    }
    out
}

/// High-precision Abramowitz & Stegun erf approximation (error < 2e-7).
pub fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x_abs);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    sign * (1.0 - poly * (-x_abs * x_abs).exp())
}

/// CRPS for Gaussian predictions on CPU.
pub fn crps_gaussian_cpu(y: &[f32], mu: &[f32], sigma: &[f32], eps: f32) -> f32 {
    let n = y.len();
    let sqrt_2 = std::f32::consts::SQRT_2;
    let inv_sqrt_pi = 1.0f32 / std::f32::consts::PI.sqrt();
    let inv_sqrt_2pi = 1.0f32 / (2.0f32 * std::f32::consts::PI).sqrt();

    let mut total = 0.0f32;
    for i in 0..n {
        let s = sigma[i].max(eps);
        let z = (y[i] - mu[i]) / s;
        let phi = (-0.5f32 * z * z).exp() * inv_sqrt_2pi;
        let big_phi = 0.5f32 * (1.0f32 + erf_approx(z / sqrt_2));
        let term = s * (z * (2.0f32 * big_phi - 1.0f32) + 2.0f32 * phi - inv_sqrt_pi);
        total += term;
    }
    total / (n as f32)
}

/// Brier Score on CPU.
pub fn brier_score_cpu(p: &[f32], y_onehot: &[f32], batch_size: usize, num_classes: usize) -> f32 {
    let mut total = 0.0f32;
    for b in 0..batch_size {
        let mut row_sum = 0.0f32;
        for c in 0..num_classes {
            let diff = p[b * num_classes + c] - y_onehot[b * num_classes + c];
            row_sum += diff * diff;
        }
        total += row_sum;
    }
    total / (batch_size as f32)
}

/// Beta-NLL on CPU.
pub fn beta_nll_cpu(y: &[f32], mu: &[f32], var: &[f32], beta: f32, eps: f32) -> f32 {
    let n = y.len();
    let mut total = 0.0f32;
    for i in 0..n {
        let v = var[i].max(eps);
        let diff = y[i] - mu[i];
        let loss = 0.5f32 * (v.ln() + (diff * diff) / v);
        let weight = v.powf(beta);
        total += weight * loss;
    }
    total / (n as f32)
}

/// Evidential Deep Learning (EDL MSE) on CPU.
pub fn edl_mse_cpu(alpha: &[f32], y_onehot: &[f32], batch_size: usize, num_classes: usize) -> f32 {
    let mut total = 0.0f32;
    for b in 0..batch_size {
        let mut s = 0.0f32;
        for c in 0..num_classes {
            s += alpha[b * num_classes + c];
        }
        let s_sq = s * s;
        let s_plus_1 = s + 1.0f32;

        let mut row_loss = 0.0f32;
        for c in 0..num_classes {
            let a = alpha[b * num_classes + c];
            let p_hat = a / s;
            let y_val = y_onehot[b * num_classes + c];
            let err_sq = (y_val - p_hat) * (y_val - p_hat);
            let var_p = (a * (s - a)) / (s_sq * s_plus_1);
            row_loss += err_sq + var_p;
        }
        total += row_loss;
    }
    total / (batch_size as f32)
}

/// ListMLE on CPU.
pub fn listmle_cpu(scores: &[f32], perm: &[i64], batch_size: usize, k_items: usize) -> f32 {
    let mut total_loss = 0.0f32;
    for b in 0..batch_size {
        let s_slice = &scores[b * k_items..(b + 1) * k_items];
        let p_slice = &perm[b * k_items..(b + 1) * k_items];

        let mut s_ordered = vec![0.0f32; k_items];
        for i in 0..k_items {
            s_ordered[i] = s_slice[p_slice[i] as usize];
        }

        for i in 0..k_items {
            let sub = &s_ordered[i..];
            let max_val = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for &val in sub {
                sum_exp += (val - max_val).exp();
            }
            let log_denom = max_val + sum_exp.ln();
            total_loss -= s_ordered[i] - log_denom;
        }
    }
    total_loss / ((batch_size * k_items) as f32)
}

/// Soft Ranks on CPU.
pub fn soft_ranks_cpu(scores: &[f32], batch_size: usize, n_items: usize, tau: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * n_items];
    for b in 0..batch_size {
        let s_slice = &scores[b * n_items..(b + 1) * n_items];
        for i in 0..n_items {
            let s_i = s_slice[i];
            let mut rank = 0.5f32;
            for j in 0..n_items {
                let s_j = s_slice[j];
                let diff = (s_i - s_j) / tau;
                let sig = 1.0f32 / (1.0f32 + (-diff).exp());
                rank += sig;
            }
            out[b * n_items + i] = rank;
        }
    }
    out
}

/// Dispatches soft-ranks evaluation to WebGPU WGSL compute shader if context available.
pub fn soft_ranks_wgpu(
    context: &crate::WgpuContext,
    scores: &[f32],
    batch_size: usize,
    n_items: usize,
    tau: f32,
) -> Option<Vec<f32>> {
    let device = &context.device;
    let queue = &context.queue;

    let total_elements = batch_size * n_items;
    let scores_bytes = bytemuck::cast_slice(scores);
    let output_size = (total_elements * std::mem::size_of::<f32>()) as u64;

    let buffer_scores = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("soft_ranks_in_scores"),
        contents: scores_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("soft_ranks_out"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("soft_ranks_staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = SoftRanksUniforms {
        batch_size: batch_size as u32,
        n_items: n_items as u32,
        tau,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("soft_ranks_uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("soft_ranks_shader"),
        source: wgpu::ShaderSource::Wgsl(SOFT_RANKS_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("soft_ranks_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("soft_ranks_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("soft_ranks_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("soft_ranks_compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("soft_ranks_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("soft_ranks_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (batch_size as u32).div_ceil(8);
        let workgroups_y = (n_items as u32).div_ceil(8);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    encoder.copy_buffer_to_buffer(&buffer_out, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device.poll(wgpu::Maintain::Wait);

    if let Ok(Ok(())) = receiver.recv() {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Some(result)
    } else {
        None
    }
}

/// Auto-dispatcher: evaluates soft ranks using WebGPU WGSL if context available, else pure-CPU.
pub fn soft_ranks_auto(
    context: Option<&crate::WgpuContext>,
    scores: &[f32],
    batch_size: usize,
    n_items: usize,
    tau: f32,
) -> Vec<f32> {
    if let Some(ctx) = context
        && let Some(res) = soft_ranks_wgpu(ctx, scores, batch_size, n_items, tau)
    {
        return res;
    }
    soft_ranks_cpu(scores, batch_size, n_items, tau)
}
