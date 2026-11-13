//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Topic 03:
//! DeepSeekMoE, Auxiliary-Loss-Free Load Balancing, and Mixture of Depths (MoD).

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for Mixture of Depths (MoD) Selective Gating:
/// Computes: y[t, d] = x[t, d] + mask[t] * gate[t] * block_out[t, d]
/// where mask[t] = 1.0 if token t is selected for execution (top-C), 0.0 otherwise.
pub const MOD_SELECTIVE_GATING_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> block_out: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    num_tokens: u32,
    d_model: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let t = global_id.x; // token index [0..num_tokens)
    let d = global_id.y; // channel index [0..d_model)

    if (t >= params.num_tokens || d >= params.d_model) {
        return;
    }

    let idx = t * params.d_model + d;
    let m = mask[t];
    let g = gate[t];
    let blk = block_out[idx];
    out_data[idx] = x[idx] + m * g * blk;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModGatingUniforms {
    num_tokens: u32,
    d_model: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Exact CPU reference implementation of MoD Selective Gating.
pub fn mod_selective_gating_cpu(
    x: &[f32],
    block_out: &[f32],
    gate: &[f32],
    mask: &[f32],
    num_tokens: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_tokens * d_model];
    for t in 0..num_tokens {
        let m = mask[t];
        let g = gate[t];
        for d in 0..d_model {
            let idx = t * d_model + d;
            out[idx] = x[idx] + m * g * block_out[idx];
        }
    }
    out
}

/// Dynamic bias update simulation on CPU: b_new = b + gamma * sign(avg_count - count).
pub fn dynamic_bias_update_cpu(b_dyn: &[f32], counts: &[f32], gamma: f32) -> Vec<f32> {
    let avg_count: f32 = counts.iter().sum::<f32>() / (counts.len() as f32);
    b_dyn
        .iter()
        .zip(counts.iter())
        .map(|(&b, &c)| {
            let diff = avg_count - c;
            let sgn = if diff > 0.0 {
                1.0
            } else if diff < 0.0 {
                -1.0
            } else {
                0.0
            };
            b + gamma * sgn
        })
        .collect()
}

/// Linear anchor forward pass on CPU: out = x @ weight + bias.
pub fn linear_anchor_cpu(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    in_cols: usize,
    out_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_cols];
    for r in 0..rows {
        for c in 0..out_cols {
            let mut sum = bias[c];
            for k in 0..in_cols {
                sum += x[r * in_cols + k] * weight[k * out_cols + c];
            }
            out[r * out_cols + c] = sum;
        }
    }
    out
}

/// WGPU compute implementation for MoD Selective Gating.
pub fn mod_selective_gating_wgpu(
    ctx: &WgpuContext,
    x: &[f32],
    block_out: &[f32],
    gate: &[f32],
    mask: &[f32],
    num_tokens: usize,
    d_model: usize,
) -> Option<Vec<f32>> {
    use wgpu::util::DeviceExt;

    let total_elements = num_tokens * d_model;
    let byte_size = (total_elements * std::mem::size_of::<f32>()) as u64;

    let x_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mod_x_buf"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let blk_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mod_block_out_buf"),
            contents: bytemuck::cast_slice(block_out),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let gate_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mod_gate_buf"),
            contents: bytemuck::cast_slice(gate),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let mask_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mod_mask_buf"),
            contents: bytemuck::cast_slice(mask),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mod_gating_out_buf"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mod_gating_readback_buf"),
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let uniforms = ModGatingUniforms {
        num_tokens: num_tokens as u32,
        d_model: d_model as u32,
        _pad0: 0,
        _pad1: 0,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mod_uniforms_buf"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mod_selective_gating_shader"),
            source: wgpu::ShaderSource::Wgsl(MOD_SELECTIVE_GATING_SHADER.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mod_gating_bgl"),
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

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mod_gating_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mod_gating_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mod_gating_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: blk_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gate_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: mask_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: u_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mod_gating_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mod_gating_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (num_tokens as u32).div_ceil(16);
        let workgroups_y = (d_model as u32).div_ceil(16);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, byte_size);
    ctx.queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    let buffer_slice = readback_buf.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });

    ctx.device.poll(wgpu::Maintain::Wait);

    if let Ok(Ok(())) = rx.recv() {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();
        Some(result)
    } else {
        None
    }
}

/// Automatically dispatches MoD Selective Gating: runs WGPU if context available, falls back to CPU.
pub fn mod_selective_gating_auto(
    ctx: Option<&WgpuContext>,
    x: &[f32],
    block_out: &[f32],
    gate: &[f32],
    mask: &[f32],
    num_tokens: usize,
    d_model: usize,
) -> Vec<f32> {
    if let Some(c) = ctx
        && let Some(gpu_out) =
            mod_selective_gating_wgpu(c, x, block_out, gate, mask, num_tokens, d_model)
    {
        return gpu_out;
    }
    mod_selective_gating_cpu(x, block_out, gate, mask, num_tokens, d_model)
}

/// Context-aware auto dispatcher for linear anchor forward pass.
pub fn linear_anchor_auto(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    in_cols: usize,
    out_cols: usize,
) -> Vec<f32> {
    linear_anchor_cpu(x, weight, bias, rows, in_cols, out_cols)
}
