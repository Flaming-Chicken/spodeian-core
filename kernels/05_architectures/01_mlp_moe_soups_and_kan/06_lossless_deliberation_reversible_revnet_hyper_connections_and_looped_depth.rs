//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Topic 06:
//! Lossless Deliberation: Multi-Lane Hyper-Connections & Reversible Coupling.

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for Hyper-Connections Multi-Lane Stream Update & Blend:
/// h1_next[i] = h1[i] + beta0 * v[i]
/// h2_next[i] = h2[i] + beta1 * v[i]
/// out_combined[i] = 0.5 * (h1_next[i] + h2_next[i])
pub const HYPER_CONNECTIONS_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> h1_in: array<f32>;
@group(0) @binding(1) var<storage, read> h2_in: array<f32>;
@group(0) @binding(2) var<storage, read> v_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> combined_out: array<f32>;

struct Uniforms {
    num_elements: u32,
    beta0: f32,
    beta1: f32,
    _pad: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    let h1_val = h1_in[idx];
    let h2_val = h2_in[idx];
    let v_val = v_in[idx];

    let h1_next = h1_val + params.beta0 * v_val;
    let h2_next = h2_val + params.beta1 * v_val;

    combined_out[idx] = 0.5 * (h1_next + h2_next);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HCUniforms {
    num_elements: u32,
    beta0: f32,
    beta1: f32,
    _pad: u32,
}

/// Exact CPU reference implementation of Hyper-Connections multi-lane update.
pub fn hyper_connections_update_cpu(
    h1: &[f32],
    h2: &[f32],
    v: &[f32],
    beta0: f32,
    beta1: f32,
) -> Vec<f32> {
    let n = h1.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let h1_next = h1[i] + beta0 * v[i];
        let h2_next = h2[i] + beta1 * v[i];
        out[i] = 0.5 * (h1_next + h2_next);
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

/// WGPU compute implementation for Hyper-Connections multi-lane update.
pub fn hyper_connections_update_wgpu(
    ctx: &WgpuContext,
    h1: &[f32],
    h2: &[f32],
    v: &[f32],
    beta0: f32,
    beta1: f32,
) -> Option<Vec<f32>> {
    use wgpu::util::DeviceExt;

    let num_elements = h1.len();
    let byte_size = (num_elements * std::mem::size_of::<f32>()) as u64;

    let h1_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hc_h1_buf"),
            contents: bytemuck::cast_slice(h1),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let h2_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hc_h2_buf"),
            contents: bytemuck::cast_slice(h2),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let v_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hc_v_buf"),
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hc_out_buf"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hc_readback_buf"),
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let uniforms = HCUniforms {
        num_elements: num_elements as u32,
        beta0,
        beta1,
        _pad: 0,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hc_uniforms_buf"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hc_update_shader"),
            source: wgpu::ShaderSource::Wgsl(HYPER_CONNECTIONS_UPDATE_SHADER.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hc_bgl"),
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

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hc_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hc_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hc_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: h1_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: h2_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: u_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hc_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hc_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (num_elements as u32 + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
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

/// Automatically dispatches Hyper-Connections multi-lane update: runs WGPU if available, falls back to CPU.
pub fn hyper_connections_update_auto(
    ctx: Option<&WgpuContext>,
    h1: &[f32],
    h2: &[f32],
    v: &[f32],
    beta0: f32,
    beta1: f32,
) -> Vec<f32> {
    if let Some(c) = ctx {
        if let Some(gpu_out) = hyper_connections_update_wgpu(c, h1, h2, v, beta0, beta1) {
            return gpu_out;
        }
    }
    hyper_connections_update_cpu(h1, h2, v, beta0, beta1)
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
