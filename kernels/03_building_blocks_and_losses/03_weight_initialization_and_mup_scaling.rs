//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 03:
//! Weight Initialization and muP Scaling (LayerScale, ReZero, Residual Scaling, and muP Readout).

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for LayerScale:
/// out[idx] = x[idx] + gamma[idx % dim] * residual[idx].
pub const LAYERSCALE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_x: array<f32>;
@group(0) @binding(1) var<storage, read> in_res: array<f32>;
@group(0) @binding(2) var<storage, read> gamma_data: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    len: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.len) {
        return;
    }
    let d = idx % params.dim;
    let x_val = in_x[idx];
    let res_val = in_res[idx];
    let g = gamma_data[d];
    out_data[idx] = x_val + g * res_val;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerScaleUniforms {
    len: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
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

/// LayerScale forward on CPU: x + gamma * residual
pub fn layerscale_cpu(
    x: &[f32],
    residual: &[f32],
    gamma: &[f32],
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * dim];
    for b in 0..batch_size {
        for i in 0..dim {
            let idx = b * dim + i;
            out[idx] = x[idx] + gamma[i] * residual[idx];
        }
    }
    out
}

/// ReZero forward on CPU: x + alpha * residual
pub fn rezero_cpu(
    x: &[f32],
    residual: &[f32],
    alpha: f32,
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * dim];
    let len = batch_size * dim;
    for i in 0..len {
        out[i] = x[i] + alpha * residual[i];
    }
    out
}

/// Residual 1/sqrt(2L) scaling forward on CPU: x + scale * residual
pub fn residual_scale_cpu(
    x: &[f32],
    residual: &[f32],
    scale: f32,
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * dim];
    let len = batch_size * dim;
    for i in 0..len {
        out[i] = x[i] + scale * residual[i];
    }
    out
}

/// muP Readout forward on CPU: (x @ weight) / width_scale
pub fn mup_readout_cpu(
    x: &[f32],
    weight: &[f32],
    width_scale: f32,
    batch_size: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * out_dim];
    let inv_scale = 1.0 / width_scale;
    for b in 0..batch_size {
        for j in 0..out_dim {
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += x[b * in_dim + i] * weight[i * out_dim + j];
            }
            out[b * out_dim + j] = sum * inv_scale;
        }
    }
    out
}

/// WebGPU kernel execution for LayerScale.
pub async fn layerscale_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &[f32],
    residual: &[f32],
    gamma: &[f32],
    batch_size: usize,
    dim: usize,
) -> Option<Vec<f32>> {
    let total_elements = (batch_size * dim) as u32;
    let byte_size = (x.len() * std::mem::size_of::<f32>()) as u64;

    let in_x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LayerScale X Buffer"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let in_res_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LayerScale Residual Buffer"),
        contents: bytemuck::cast_slice(residual),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let gamma_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LayerScale Gamma Buffer"),
        contents: bytemuck::cast_slice(gamma),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("LayerScale Output Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("LayerScale Staging Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = LayerScaleUniforms {
        len: total_elements,
        dim: dim as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LayerScale Uniform Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("LayerScale Shader"),
        source: wgpu::ShaderSource::Wgsl(LAYERSCALE_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("LayerScale Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
        label: Some("LayerScale Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("LayerScale Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("LayerScale Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in_res_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gamma_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("LayerScale Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LayerScale Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (total_elements + 63) / 64;
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

/// Context-aware auto dispatcher for LayerScale.
pub async fn layerscale_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    x: &[f32],
    residual: &[f32],
    gamma: &[f32],
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    if let Some(ctx) = context {
        if let Some(res) =
            layerscale_wgpu(&ctx.0, &ctx.1, x, residual, gamma, batch_size, dim).await
        {
            return res;
        }
    }
    layerscale_cpu(x, residual, gamma, batch_size, dim)
}
