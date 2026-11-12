//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 04:
//! Positional and Structural Encodings (Sinusoidal, RoPE, ALiBi, and RWPE).

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for Rotary Position Embedding (RoPE) 2D pair rotation.
pub const ROPE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read> cos_data: array<f32>;
@group(0) @binding(2) var<storage, read> sin_data: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    seq_len: u32,
    half_dim: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let total_pairs = params.seq_len * params.half_dim;
    if (pair_idx >= total_pairs) {
        return;
    }
    let s = pair_idx / params.half_dim;
    let i = pair_idx % params.half_dim;
    let in_offset = s * (params.half_dim * 2u);
    let x1 = in_data[in_offset + 2u * i];
    let x2 = in_data[in_offset + 2u * i + 1u];
    let c = cos_data[s * params.half_dim + i];
    let sn = sin_data[s * params.half_dim + i];

    out_data[in_offset + 2u * i] = x1 * c - x2 * sn;
    out_data[in_offset + 2u * i + 1u] = x1 * sn + x2 * c;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RoPEUniforms {
    seq_len: u32,
    half_dim: u32,
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

/// Sinusoidal Positional Encoding forward on CPU: x + pe[0..seq_len*dim]
pub fn sinusoidal_pe_cpu(x: &[f32], pe: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * dim];
    for i in 0..seq_len * dim {
        out[i] = x[i] + pe[i];
    }
    out
}

/// RoPE forward on CPU.
pub fn rope_cpu(x: &[f32], cos: &[f32], sin: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * dim];
    let half_dim = dim / 2;
    for s in 0..seq_len {
        for i in 0..half_dim {
            let x1 = x[s * dim + 2 * i];
            let x2 = x[s * dim + 2 * i + 1];
            let c = cos[s * half_dim + i];
            let sn = sin[s * half_dim + i];
            out[s * dim + 2 * i] = x1 * c - x2 * sn;
            out[s * dim + 2 * i + 1] = x1 * sn + x2 * c;
        }
    }
    out
}

/// ALiBi forward on CPU: -slopes[h] * |i - j|
pub fn alibi_cpu(seq_len: usize, slopes: &[f32], num_heads: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * seq_len * seq_len];
    for h in 0..num_heads {
        let m = slopes[h];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dist = ((i as isize) - (j as isize)).abs() as f32;
                let idx = h * seq_len * seq_len + i * seq_len + j;
                out[idx] = -m * dist;
            }
        }
    }
    out
}

/// RWPE forward on CPU: landing probabilities P^k_ii for k in 1..k_steps
pub fn rwpe_cpu(adj: &[f32], num_nodes: usize, k_steps: usize) -> Vec<f32> {
    // Degree vector
    let mut deg = vec![0.0f32; num_nodes];
    for i in 0..num_nodes {
        let mut d = 0.0f32;
        for j in 0..num_nodes {
            d += adj[i * num_nodes + j];
        }
        deg[i] = d;
    }

    // Transition matrix P = D^-1 A
    let mut p_walk = vec![0.0f32; num_nodes * num_nodes];
    for i in 0..num_nodes {
        let d = deg[i];
        if d > 0.0 {
            for j in 0..num_nodes {
                p_walk[i * num_nodes + j] = adj[i * num_nodes + j] / d;
            }
        }
    }

    // P_curr = I
    let mut p_curr = vec![0.0f32; num_nodes * num_nodes];
    for i in 0..num_nodes {
        p_curr[i * num_nodes + i] = 1.0;
    }

    // Output rwpe: [num_nodes, k_steps]
    let mut rwpe = vec![0.0f32; num_nodes * k_steps];
    for k in 0..k_steps {
        let mut p_next = vec![0.0f32; num_nodes * num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let mut sum = 0.0f32;
                for m in 0..num_nodes {
                    sum += p_curr[i * num_nodes + m] * p_walk[m * num_nodes + j];
                }
                p_next[i * num_nodes + j] = sum;
            }
            rwpe[i * k_steps + k] = p_next[i * num_nodes + i];
        }
        p_curr = p_next;
    }

    rwpe
}

/// WebGPU kernel execution for RoPE.
pub async fn rope_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    seq_len: usize,
    dim: usize,
) -> Option<Vec<f32>> {
    let half_dim = (dim / 2) as u32;
    let total_pairs = (seq_len as u32) * half_dim;
    let byte_size = (x.len() * std::mem::size_of::<f32>()) as u64;
    let _angle_byte_size = (cos.len() * std::mem::size_of::<f32>()) as u64;

    let in_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RoPE Input Buffer"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let cos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RoPE Cos Buffer"),
        contents: bytemuck::cast_slice(cos),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sin_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RoPE Sin Buffer"),
        contents: bytemuck::cast_slice(sin),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RoPE Output Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RoPE Staging Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = RoPEUniforms {
        seq_len: seq_len as u32,
        half_dim,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RoPE Uniform Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("RoPE Shader"),
        source: wgpu::ShaderSource::Wgsl(ROPE_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("RoPE Bind Group Layout"),
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
        label: Some("RoPE Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("RoPE Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("RoPE Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cos_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sin_buffer.as_entire_binding(),
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
        label: Some("RoPE Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RoPE Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (total_pairs + 63) / 64;
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

/// Context-aware auto dispatcher for RoPE.
pub async fn rope_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    if let Some(ctx) = context {
        if let Some(res) = rope_wgpu(&ctx.0, &ctx.1, x, cos, sin, seq_len, dim).await {
            return res;
        }
    }
    rope_cpu(x, cos, sin, seq_len, dim)
}
