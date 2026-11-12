//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Topic 04:
//! Soup of Experts (SMEAR: Dynamic Parameter Merging in Weight Space).

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for Dynamic Parameter Merging (Soup of Experts / SMEAR):
/// Computes: merged_w[i, j] = sum_{e=0}^{E-1} g[e] * expert_w[e, i, j]
pub const SOUP_WEIGHT_MERGE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> expert_weights: array<f32>;
@group(0) @binding(1) var<storage, read> routing_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> merged_weights: array<f32>;

struct Uniforms {
    num_experts: u32,
    in_dim: u32,
    out_dim: u32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x; // row index [0..in_dim)
    let j = global_id.y; // col index [0..out_dim)

    if (i >= params.in_dim || j >= params.out_dim) {
        return;
    }

    var sum: f32 = 0.0;
    let plane_size = params.in_dim * params.out_dim;
    for (var e: u32 = 0u; e < params.num_experts; e = e + 1u) {
        let g = routing_weights[e];
        let idx = e * plane_size + i * params.out_dim + j;
        sum = sum + g * expert_weights[idx];
    }

    let out_idx = i * params.out_dim + j;
    merged_weights[out_idx] = sum;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SoupMergeUniforms {
    num_experts: u32,
    in_dim: u32,
    out_dim: u32,
    _pad: u32,
}

/// Exact CPU reference implementation of Dynamic Parameter Merging.
pub fn soup_weight_merge_cpu(
    expert_weights: &[f32],
    routing_weights: &[f32],
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut merged = vec![0.0f32; in_dim * out_dim];
    let plane_size = in_dim * out_dim;

    for i in 0..in_dim {
        for j in 0..out_dim {
            let mut sum = 0.0f32;
            for e in 0..num_experts {
                let g = routing_weights[e];
                let idx = e * plane_size + i * out_dim + j;
                sum += g * expert_weights[idx];
            }
            merged[i * out_dim + j] = sum;
        }
    }
    merged
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

/// WGPU compute implementation for Dynamic Parameter Merging.
pub fn soup_weight_merge_wgpu(
    ctx: &WgpuContext,
    expert_weights: &[f32],
    routing_weights: &[f32],
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
) -> Option<Vec<f32>> {
    use wgpu::util::DeviceExt;

    let total_elements = in_dim * out_dim;
    let byte_size = (total_elements * std::mem::size_of::<f32>()) as u64;

    let ew_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("soup_expert_weights_buf"),
            contents: bytemuck::cast_slice(expert_weights),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let rw_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("soup_routing_weights_buf"),
            contents: bytemuck::cast_slice(routing_weights),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("soup_merged_weights_out_buf"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("soup_merged_weights_readback_buf"),
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let uniforms = SoupMergeUniforms {
        num_experts: num_experts as u32,
        in_dim: in_dim as u32,
        out_dim: out_dim as u32,
        _pad: 0,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("soup_merge_uniforms_buf"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soup_weight_merge_shader"),
            source: wgpu::ShaderSource::Wgsl(SOUP_WEIGHT_MERGE_SHADER.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soup_merge_bgl"),
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
            label: Some("soup_merge_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soup_merge_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("soup_merge_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ew_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rw_buf.as_entire_binding(),
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
            label: Some("soup_merge_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("soup_merge_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (in_dim as u32 + 15) / 16;
        let workgroups_y = (out_dim as u32 + 15) / 16;
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

/// Automatically dispatches Parameter Merging: runs WGPU if context available, falls back to CPU.
pub fn soup_weight_merge_auto(
    ctx: Option<&WgpuContext>,
    expert_weights: &[f32],
    routing_weights: &[f32],
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    if let Some(c) = ctx {
        if let Some(gpu_out) = soup_weight_merge_wgpu(
            c,
            expert_weights,
            routing_weights,
            num_experts,
            in_dim,
            out_dim,
        ) {
            return gpu_out;
        }
    }
    soup_weight_merge_cpu(
        expert_weights,
        routing_weights,
        num_experts,
        in_dim,
        out_dim,
    )
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
