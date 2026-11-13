//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 06:
//! Regularization and Training Stabilizers (Dropout, DropPath, Soft-Capping, Label Smoothing, and Router Z-Loss).

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for Logit Soft-Capping: C * tanh(x / C).
pub const SOFTCAP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    total_elements: u32,
    cap: f32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    let scaled = in_data[idx] / params.cap;
    out_data[idx] = params.cap * tanh(scaled);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SoftCapUniforms {
    total_elements: u32,
    cap: f32,
    _pad0: u32,
    _pad1: u32,
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

/// Inverted Dropout forward on CPU: (x * mask) / (1 - p).
pub fn dropout_cpu(x: &[f32], mask: &[f32], p: f32) -> Vec<f32> {
    let scale = 1.0 / (1.0 - p);
    x.iter()
        .zip(mask.iter())
        .map(|(&xi, &mi)| xi * mi * scale)
        .collect()
}

/// DropPath forward on CPU: x_identity + (x_branch * mask) / (1 - drop_prob).
pub fn droppath_cpu(
    x_identity: &[f32],
    x_branch: &[f32],
    mask: &[f32],
    b: usize,
    d: usize,
    drop_prob: f32,
) -> Vec<f32> {
    let scale = 1.0 / (1.0 - drop_prob);
    let mut out = vec![0.0f32; b * d];
    for bi in 0..b {
        let m = mask[bi];
        for di in 0..d {
            let idx = bi * d + di;
            out[idx] = x_identity[idx] + (x_branch[idx] * m) * scale;
        }
    }
    out
}

/// Logit Soft-Capping forward on CPU: cap * tanh(x / cap).
pub fn softcapping_cpu(x: &[f32], cap: f32) -> Vec<f32> {
    x.iter().map(|&xi| cap * (xi / cap).tanh()).collect()
}

/// Label Smoothing Cross-Entropy Loss on CPU.
pub fn label_smoothing_loss_cpu(
    logits: &[f32],
    targets: &[f32],
    b: usize,
    k: usize,
    alpha: f32,
) -> f32 {
    let mut total_loss = 0.0f32;
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
        let nll = -(logits[base + target_idx] - lse);

        let mut uniform_loss = 0.0f32;
        for c in 0..k {
            uniform_loss += -(logits[base + c] - lse);
        }
        uniform_loss /= k as f32;

        total_loss += (1.0 - alpha) * nll + alpha * uniform_loss;
    }
    total_loss / (b as f32)
}

/// Router Z-Loss forward on CPU: (z_coef / B) * sum( (log sum exp(z))^2 ).
pub fn router_z_loss_cpu(router_logits: &[f32], b: usize, k: usize, z_coef: f32) -> f32 {
    let mut sum_sq_lse = 0.0f32;
    for bi in 0..b {
        let base = bi * k;
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..k {
            let val = router_logits[base + c];
            if val > max_val {
                max_val = val;
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..k {
            sum_exp += (router_logits[base + c] - max_val).exp();
        }
        let lse = max_val + sum_exp.ln();
        sum_sq_lse += lse.powi(2);
    }
    (z_coef / (b as f32)) * sum_sq_lse
}

/// WGPU compute implementation for Logit Soft-Capping.
pub async fn softcapping_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &[f32],
    cap: f32,
) -> Option<Vec<f32>> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SoftCap Shader"),
        source: wgpu::ShaderSource::Wgsl(SOFTCAP_SHADER.into()),
    });

    let total_elements = x.len() as u32;
    let byte_size = std::mem::size_of_val(x) as u64;

    let in_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SoftCap In Buffer"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SoftCap Out Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SoftCap Staging Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = SoftCapUniforms {
        total_elements,
        cap,
        _pad0: 0,
        _pad1: 0,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SoftCap Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SoftCap Bind Group Layout"),
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
        label: Some("SoftCap Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SoftCap Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SoftCap Bind Group"),
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
        label: Some("SoftCap Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SoftCap Pass"),
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

/// Context-aware auto dispatcher for Logit Soft-Capping.
pub async fn softcapping_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    x: &[f32],
    cap: f32,
) -> Vec<f32> {
    if let Some(ctx) = context
        && let Some(res) = softcapping_wgpu(&ctx.0, &ctx.1, x, cap).await
    {
        return res;
    }
    softcapping_cpu(x, cap)
}
