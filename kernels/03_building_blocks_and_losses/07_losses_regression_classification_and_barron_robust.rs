//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 07:
//! Losses: Regression, Classification, and Barron Adaptive Robust.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for Barron's General Adaptive Robust Loss.
pub const BARRON_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_residuals: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_loss: array<f32>;

struct Uniforms {
    total_elements: u32,
    alpha: f32,
    c: f32,
    d_alpha: f32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    let r = in_residuals[idx];
    let scaled_sq = (r / params.c) * (r / params.c);
    let inner = scaled_sq / params.d_alpha + 1.0;
    let pow_inner = pow(inner, params.alpha * 0.5);
    out_loss[idx] = (params.d_alpha / params.alpha) * (pow_inner - 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BarronUniforms {
    total_elements: u32,
    alpha: f32,
    c: f32,
    d_alpha: f32,
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

/// Barron's General Adaptive Robust Loss forward on CPU.
pub fn barron_robust_loss_cpu(r: &[f32], alpha: f32, c: f32) -> f32 {
    let d_alpha = (alpha - 2.0).abs();
    let mut total = 0.0f32;
    for &val in r {
        let scaled_sq = (val / c).powi(2);
        let inner = scaled_sq / d_alpha + 1.0;
        let pow_inner = inner.powf(alpha * 0.5);
        total += (d_alpha / alpha) * (pow_inner - 1.0);
    }
    total / (r.len() as f32)
}

/// Huber Loss forward on CPU (Smooth L1 piecewise regression).
pub fn huber_loss_cpu(r: &[f32], delta: f32) -> f32 {
    let mut total = 0.0f32;
    for &val in r {
        let abs_v = val.abs();
        if abs_v <= delta {
            total += 0.5 * val.powi(2);
        } else {
            total += delta * (abs_v - 0.5 * delta);
        }
    }
    total / (r.len() as f32)
}

/// Multi-Class Focal Loss forward on CPU: -(1 - p_t)^gamma * log(p_t).
pub fn focal_loss_cpu(logits: &[f32], targets: &[f32], b: usize, k: usize, gamma: f32) -> f32 {
    let mut total = 0.0f32;
    for bi in 0..b {
        let base = bi * k;
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..k {
            let val = logits[base + c];
            if val > max_val {
                max_val = val;
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..k {
            sum_exp += (logits[base + c] - max_val).exp();
        }
        let lse = max_val + sum_exp.ln();

        let target_idx = targets[bi] as usize;
        let log_p_t = logits[base + target_idx] - lse;
        let p_t = log_p_t.exp();

        total += -(1.0 - p_t).powf(gamma) * log_p_t;
    }
    total / (b as f32)
}

/// Poly-1 Loss forward on CPU: -log(p_t) + epsilon_1 * (1 - p_t).
pub fn polyloss_cpu(logits: &[f32], targets: &[f32], b: usize, k: usize, epsilon_1: f32) -> f32 {
    let mut total = 0.0f32;
    for bi in 0..b {
        let base = bi * k;
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..k {
            let val = logits[base + c];
            if val > max_val {
                max_val = val;
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..k {
            sum_exp += (logits[base + c] - max_val).exp();
        }
        let lse = max_val + sum_exp.ln();

        let target_idx = targets[bi] as usize;
        let log_p_t = logits[base + target_idx] - lse;
        let p_t = log_p_t.exp();

        total += -log_p_t + epsilon_1 * (1.0 - p_t);
    }
    total / (b as f32)
}

/// Asymmetric Loss (ASL) forward on CPU for multi-label classification.
pub fn asymmetric_loss_cpu(
    logits: &[f32],
    targets: &[f32],
    gamma_pos: f32,
    gamma_neg: f32,
    clip_margin: f32,
    eps: f32,
) -> f32 {
    let mut total = 0.0f32;
    let n = logits.len();
    for i in 0..n {
        let z = logits[i];
        let y = targets[i];
        let p = 1.0 / (1.0 + (-z).exp());
        let p_m = (p - clip_margin).max(0.0);

        let pos_loss = -y * (1.0 - p).powf(gamma_pos) * p.max(eps).ln();
        let neg_loss = -(1.0 - y) * p_m.powf(gamma_neg) * (1.0 - p_m).max(eps).ln();
        total += pos_loss + neg_loss;
    }
    total / (n as f32)
}

/// WGPU compute implementation for Barron Robust Loss elementwise evaluations.
pub async fn barron_robust_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    r: &[f32],
    alpha: f32,
    c: f32,
) -> Option<Vec<f32>> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Barron Shader"),
        source: wgpu::ShaderSource::Wgsl(BARRON_SHADER.into()),
    });

    let total_elements = r.len() as u32;
    let byte_size = std::mem::size_of_val(r) as u64;

    let in_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Barron In Buffer"),
        contents: bytemuck::cast_slice(r),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Barron Out Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Barron Staging Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = BarronUniforms {
        total_elements,
        alpha,
        c,
        d_alpha: (alpha - 2.0).abs(),
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Barron Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Barron Bind Group Layout"),
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Barron Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Barron Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Barron Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Barron Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Barron Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = total_elements.div_ceil(64);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging_buffer, 0, byte_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = sender.send(v);
    });

    let _ = device.poll(wgpu::MaintainBase::Wait);
    receiver.recv().ok()?.ok()?;

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    Some(result)
}

/// Context-aware auto dispatcher for Barron Robust Loss.
pub async fn barron_robust_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    r: &[f32],
    alpha: f32,
    c: f32,
) -> f32 {
    if let Some(ctx) = context
        && let Some(res) = barron_robust_wgpu(&ctx.0, &ctx.1, r, alpha, c).await
    {
        let sum: f32 = res.iter().sum();
        return sum / (res.len() as f32);
    }
    barron_robust_loss_cpu(r, alpha, c)
}
