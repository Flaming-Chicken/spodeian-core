//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Topic 02:
//! Mixture of Experts (Soft MoE Dispatch and Combine Operators).

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for Soft MoE Dispatch:
/// Computes slot inputs: slot_in[j, d] = sum_{t=0}^{T-1} phi[t, j] * x[t, d]
pub const SOFT_MOE_DISPATCH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> phi: array<f32>;
@group(0) @binding(2) var<storage, read_write> slot_inputs: array<f32>;

struct Uniforms {
    num_tokens: u32,
    d_model: u32,
    total_slots: u32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x; // slot index [0..total_slots)
    let d = global_id.y; // channel index [0..d_model)

    if (j >= params.total_slots || d >= params.d_model) {
        return;
    }

    var sum: f32 = 0.0;
    for (var t: u32 = 0u; t < params.num_tokens; t = t + 1u) {
        let phi_val = phi[t * params.total_slots + j];
        let x_val = x[t * params.d_model + d];
        sum = sum + phi_val * x_val;
    }

    let out_idx = j * params.d_model + d;
    slot_inputs[out_idx] = sum;
}
"#;

/// WGSL compute shader for Soft MoE Combine:
/// Computes token outputs: y[t, d] = sum_{j=0}^{M-1} psi[t, j] * slot_out[j, d]
pub const SOFT_MOE_COMBINE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> psi: array<f32>;
@group(0) @binding(1) var<storage, read> slot_outputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    num_tokens: u32,
    d_model: u32,
    total_slots: u32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let t = global_id.x; // token index [0..num_tokens)
    let d = global_id.y; // channel index [0..d_model)

    if (t >= params.num_tokens || d >= params.d_model) {
        return;
    }

    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < params.total_slots; j = j + 1u) {
        let psi_val = psi[t * params.total_slots + j];
        let slot_val = slot_outputs[j * params.d_model + d];
        sum = sum + psi_val * slot_val;
    }

    let out_idx = t * params.d_model + d;
    out_data[out_idx] = sum;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SoftMoeUniforms {
    num_tokens: u32,
    d_model: u32,
    total_slots: u32,
    _pad: u32,
}

/// Exact CPU reference implementation of Soft MoE Dispatch: slot_in = Phi^T @ x
pub fn soft_moe_dispatch_cpu(
    x: &[f32],
    phi: &[f32],
    num_tokens: usize,
    d_model: usize,
    total_slots: usize,
) -> Vec<f32> {
    let mut slot_inputs = vec![0.0f32; total_slots * d_model];
    for j in 0..total_slots {
        for d in 0..d_model {
            let mut sum = 0.0f32;
            for t in 0..num_tokens {
                let phi_val = phi[t * total_slots + j];
                let x_val = x[t * d_model + d];
                sum += phi_val * x_val;
            }
            slot_inputs[j * d_model + d] = sum;
        }
    }
    slot_inputs
}

/// Exact CPU reference implementation of Soft MoE Combine: out = Psi @ slot_outputs
pub fn soft_moe_combine_cpu(
    psi: &[f32],
    slot_outputs: &[f32],
    num_tokens: usize,
    d_model: usize,
    total_slots: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_tokens * d_model];
    for t in 0..num_tokens {
        for d in 0..d_model {
            let mut sum = 0.0f32;
            for j in 0..total_slots {
                let psi_val = psi[t * total_slots + j];
                let slot_val = slot_outputs[j * d_model + d];
                sum += psi_val * slot_val;
            }
            out[t * d_model + d] = sum;
        }
    }
    out
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

/// WGPU compute implementation for Soft MoE Dispatch.
pub fn soft_moe_dispatch_wgpu(
    ctx: &WgpuContext,
    x: &[f32],
    phi: &[f32],
    num_tokens: usize,
    d_model: usize,
    total_slots: usize,
) -> Option<Vec<f32>> {
    use wgpu::util::DeviceExt;

    let total_output_elements = total_slots * d_model;
    let output_byte_size = (total_output_elements * std::mem::size_of::<f32>()) as u64;

    let x_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moe_x_buf"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let phi_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moe_phi_buf"),
            contents: bytemuck::cast_slice(phi),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("moe_dispatch_out_buf"),
        size: output_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("moe_dispatch_readback_buf"),
        size: output_byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let uniforms = SoftMoeUniforms {
        num_tokens: num_tokens as u32,
        d_model: d_model as u32,
        total_slots: total_slots as u32,
        _pad: 0,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moe_uniforms_buf"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soft_moe_dispatch_shader"),
            source: wgpu::ShaderSource::Wgsl(SOFT_MOE_DISPATCH_SHADER.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soft_moe_dispatch_bgl"),
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

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soft_moe_dispatch_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soft_moe_dispatch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("soft_moe_dispatch_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: phi_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: u_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("soft_moe_dispatch_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("soft_moe_dispatch_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (total_slots as u32 + 15) / 16;
        let workgroups_y = (d_model as u32 + 15) / 16;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, output_byte_size);
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

/// Automatically dispatches Soft MoE Dispatch: runs WGPU if context available, falls back to CPU.
pub fn soft_moe_dispatch_auto(
    ctx: Option<&WgpuContext>,
    x: &[f32],
    phi: &[f32],
    num_tokens: usize,
    d_model: usize,
    total_slots: usize,
) -> Vec<f32> {
    if let Some(c) = ctx {
        if let Some(gpu_out) = soft_moe_dispatch_wgpu(c, x, phi, num_tokens, d_model, total_slots) {
            return gpu_out;
        }
    }
    soft_moe_dispatch_cpu(x, phi, num_tokens, d_model, total_slots)
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
