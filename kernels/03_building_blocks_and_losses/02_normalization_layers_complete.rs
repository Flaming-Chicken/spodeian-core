//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 02:
//! Normalization Layers Complete (LayerNorm, RMSNorm, DyT, adaLN-Zero, and Weight Standardization).

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for Dynamic Tanh (DyT) reduction-free normalization:
/// DyT(x) = gamma * tanh(alpha * x) + beta.
pub const DYT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read> alpha_data: array<f32>;
@group(0) @binding(2) var<storage, read> gamma_data: array<f32>;
@group(0) @binding(3) var<storage, read> beta_data: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    len: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.len) {
        return;
    }
    let d = idx % params.dim;
    let x = in_data[idx];
    let a = alpha_data[d];
    let g = gamma_data[d];
    let b = beta_data[d];
    out_data[idx] = g * tanh(a * x) + b;
}
"#;

/// WGSL compute shader for Root Mean Square Normalization (RMSNorm):
/// y = (x / sqrt(mean(x^2) + eps)) * gamma.
pub const RMSNORM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read> gamma_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    batch_size: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.batch_size) {
        return;
    }
    let offset = row * params.dim;
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.dim; i = i + 1u) {
        let v = in_data[offset + i];
        sum_sq = sum_sq + v * v;
    }
    let rms = sqrt(sum_sq / f32(params.dim) + params.eps);
    for (var i: u32 = 0u; i < params.dim; i = i + 1u) {
        let v = in_data[offset + i];
        let g = gamma_data[i];
        out_data[offset + i] = (v / rms) * g;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DyTUniforms {
    len: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RMSNormUniforms {
    batch_size: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}

/// Linear anchor forward on CPU.
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

/// LayerNorm forward on CPU: ((x - mean) / sqrt(var + eps)) * gamma + beta
pub fn layernorm_cpu(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * dim];
    for b in 0..batch_size {
        let row = &x[b * dim..(b + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..dim {
            out[b * dim + i] = ((row[i] - mean) * inv_std) * gamma[i] + beta[i];
        }
    }
    out
}

/// RMSNorm forward on CPU: (x / sqrt(mean(x^2) + eps)) * gamma
pub fn rmsnorm_cpu(x: &[f32], gamma: &[f32], batch_size: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * dim];
    for b in 0..batch_size {
        let row = &x[b * dim..(b + 1) * dim];
        let sum_sq = row.iter().map(|&v| v * v).sum::<f32>();
        let rms = (sum_sq / dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..dim {
            out[b * dim + i] = (row[i] * inv_rms) * gamma[i];
        }
    }
    out
}

/// Dynamic Tanh (DyT) forward on CPU: gamma * tanh(alpha * x) + beta (reduction-free)
pub fn dyt_cpu(
    x: &[f32],
    alpha: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * dim];
    for b in 0..batch_size {
        for i in 0..dim {
            let idx = b * dim + i;
            let val = x[idx];
            out[idx] = gamma[i] * (alpha[i] * val).tanh() + beta[i];
        }
    }
    out
}

/// adaLN-Zero forward on CPU: (1 + gamma_mod) * LayerNorm(x) + beta_mod
pub fn adaln_zero_cpu(
    x: &[f32],
    cond: &[f32],
    w_adaln: &[f32],
    b_adaln: &[f32],
    batch_size: usize,
    dim: usize,
    cond_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let base_ln = layernorm_cpu(
        x,
        &vec![1.0f32; dim],
        &vec![0.0f32; dim],
        batch_size,
        dim,
        eps,
    );

    let mod_dim = 2 * dim;
    let mut proj = vec![0.0f32; batch_size * mod_dim];
    for b in 0..batch_size {
        for j in 0..mod_dim {
            let mut sum = b_adaln[j];
            for k in 0..cond_dim {
                sum += cond[b * cond_dim + k] * w_adaln[k * mod_dim + j];
            }
            proj[b * mod_dim + j] = sum;
        }
    }

    let mut out = vec![0.0f32; batch_size * dim];
    for b in 0..batch_size {
        for i in 0..dim {
            let gamma_mod = proj[b * mod_dim + i];
            let beta_mod = proj[b * mod_dim + dim + i];
            out[b * dim + i] = (1.0 + gamma_mod) * base_ln[b * dim + i] + beta_mod;
        }
    }
    out
}

/// QK-Norm forward on CPU: x / sqrt(mean(x^2) + eps)
pub fn qknorm_cpu(x: &[f32], batch_size: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * dim];
    for b in 0..batch_size {
        let row = &x[b * dim..(b + 1) * dim];
        let sum_sq = row.iter().map(|&v| v * v).sum::<f32>();
        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        for i in 0..dim {
            out[b * dim + i] = row[i] * inv_rms;
        }
    }
    out
}

/// Scaled Weight Standardization along fan-in dimension (cols) on CPU
pub fn scaled_weight_standardization_cpu(
    w: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    let fan_in_scale = (cols as f32).sqrt();
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let denom = (var + eps).sqrt() * fan_in_scale;
        let inv_denom = 1.0 / denom;
        for c in 0..cols {
            out[r * cols + c] = (row[c] - mean) * inv_denom;
        }
    }
    out
}

/// WebGPU kernel execution for Dynamic Tanh (DyT).
pub async fn dyt_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &[f32],
    alpha: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    dim: usize,
) -> Option<Vec<f32>> {
    let total_elements = (batch_size * dim) as u32;
    let byte_size = std::mem::size_of_val(x) as u64;

    let in_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("DyT Input Buffer"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let alpha_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("DyT Alpha Buffer"),
        contents: bytemuck::cast_slice(alpha),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let gamma_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("DyT Gamma Buffer"),
        contents: bytemuck::cast_slice(gamma),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let beta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("DyT Beta Buffer"),
        contents: bytemuck::cast_slice(beta),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("DyT Output Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("DyT Staging Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = DyTUniforms {
        len: total_elements,
        dim: dim as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("DyT Uniform Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("DyT Shader"),
        source: wgpu::ShaderSource::Wgsl(DYT_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("DyT Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
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
        label: Some("DyT Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("DyT Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DyT Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: alpha_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gamma_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: beta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("DyT Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("DyT Pass"),
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

/// WebGPU kernel execution for RMSNorm.
pub async fn rmsnorm_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &[f32],
    gamma: &[f32],
    batch_size: usize,
    dim: usize,
    eps: f32,
) -> Option<Vec<f32>> {
    let byte_size = std::mem::size_of_val(x) as u64;

    let in_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RMSNorm Input Buffer"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let gamma_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RMSNorm Gamma Buffer"),
        contents: bytemuck::cast_slice(gamma),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RMSNorm Output Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RMSNorm Staging Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = RMSNormUniforms {
        batch_size: batch_size as u32,
        dim: dim as u32,
        eps,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RMSNorm Uniform Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("RMSNorm Shader"),
        source: wgpu::ShaderSource::Wgsl(RMSNORM_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("RMSNorm Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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
        label: Some("RMSNorm Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("RMSNorm Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("RMSNorm Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gamma_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("RMSNorm Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RMSNorm Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (batch_size as u32).div_ceil(64);
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

/// Context-aware auto dispatcher for Dynamic Tanh (DyT).
pub async fn dyt_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    x: &[f32],
    alpha: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    if let Some(ctx) = context
        && let Some(res) = dyt_wgpu(&ctx.0, &ctx.1, x, alpha, gamma, beta, batch_size, dim).await
    {
        return res;
    }
    dyt_cpu(x, alpha, gamma, beta, batch_size, dim)
}

/// Context-aware auto dispatcher for RMSNorm.
pub async fn rmsnorm_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    x: &[f32],
    gamma: &[f32],
    batch_size: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    if let Some(ctx) = context
        && let Some(res) = rmsnorm_wgpu(&ctx.0, &ctx.1, x, gamma, batch_size, dim, eps).await
    {
        return res;
    }
    rmsnorm_cpu(x, gamma, batch_size, dim, eps)
}
