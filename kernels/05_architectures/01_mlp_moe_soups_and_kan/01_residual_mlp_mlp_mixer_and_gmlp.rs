//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Topic 01:
//! Residual MLP, MLP-Mixer, and gMLP (Spatial Gating Unit & Token/Channel Mixing).

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for gMLP Spatial Gating Unit:
/// Computes gate_s = (sum_k W_sgu[s, k] * Z2[b, k, d]) + b_sgu[s]
/// and outputs Z_out[b, s, d] = Z1[b, s, d] * gate_s
pub const SGU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> z1: array<f32>;
@group(0) @binding(1) var<storage, read> z2: array<f32>;
@group(0) @binding(2) var<storage, read> w_sgu: array<f32>;
@group(0) @binding(3) var<storage, read> b_sgu: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    batch_size: u32,
    seq_len: u32,
    dim: u32,
    _pad: u32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = global_id.x; // sequence token index [0..seq_len)
    let d = global_id.y; // channel dimension index [0..dim)
    let b = global_id.z; // batch index [0..batch_size)

    if (s >= params.seq_len || d >= params.dim || b >= params.batch_size) {
        return;
    }

    var gate: f32 = b_sgu[s];
    for (var k: u32 = 0u; k < params.seq_len; k = k + 1u) {
        let w_val = w_sgu[s * params.seq_len + k];
        let z2_idx = b * (params.seq_len * params.dim) + k * params.dim + d;
        gate = gate + w_val * z2[z2_idx];
    }

    let out_idx = b * (params.seq_len * params.dim) + s * params.dim + d;
    out_data[out_idx] = z1[out_idx] * gate;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SguUniforms {
    batch_size: u32,
    seq_len: u32,
    dim: u32,
    _pad: u32,
}

/// Exact CPU reference implementation of Spatial Gating Unit (SGU).
pub fn spatial_gating_cpu(
    z1: &[f32],
    z2: &[f32],
    w_sgu: &[f32],
    b_sgu: &[f32],
    batch_size: usize,
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    let total_elements = batch_size * seq_len * dim;
    let mut out = vec![0.0f32; total_elements];

    for b in 0..batch_size {
        for s in 0..seq_len {
            for d in 0..dim {
                let mut gate = b_sgu[s];
                for k in 0..seq_len {
                    let w_val = w_sgu[s * seq_len + k];
                    let z2_idx = b * (seq_len * dim) + k * dim + d;
                    gate += w_val * z2[z2_idx];
                }
                let idx = b * (seq_len * dim) + s * dim + d;
                out[idx] = z1[idx] * gate;
            }
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

/// WGPU kernel dispatcher for Spatial Gating Unit.
#[cfg(feature = "wgpu")]
pub async fn spatial_gating_wgpu(
    ctx: &WgpuContext,
    z1: &[f32],
    z2: &[f32],
    w_sgu: &[f32],
    b_sgu: &[f32],
    batch_size: usize,
    seq_len: usize,
    dim: usize,
) -> Result<Vec<f32>, String> {
    use wgpu::util::DeviceExt;

    let total_elements = batch_size * seq_len * dim;
    let size_bytes = (total_elements * std::mem::size_of::<f32>()) as u64;

    let uniforms = SguUniforms {
        batch_size: batch_size as u32,
        seq_len: seq_len as u32,
        dim: dim as u32,
        _pad: 0,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sgu_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let z1_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sgu_z1"),
            contents: bytemuck::cast_slice(z1),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let z2_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sgu_z2"),
            contents: bytemuck::cast_slice(z2),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let w_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sgu_w"),
            contents: bytemuck::cast_slice(w_sgu),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let b_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sgu_b"),
            contents: bytemuck::cast_slice(b_sgu),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sgu_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sgu_shader"),
            source: wgpu::ShaderSource::Wgsl(SGU_SHADER.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sgu_bgl"),
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
            label: Some("sgu_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sgu_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sgu_bind_group"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: z1_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: z2_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_buf.as_entire_binding(),
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
            label: Some("sgu_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sgu_cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (seq_len as u32).div_ceil(16);
        let wg_y = (dim as u32).div_ceil(16);
        let wg_z = batch_size as u32;
        cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }

    let stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sgu_stage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, size_bytes);
    ctx.queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    let slice = stage.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());

    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;

    let res = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
    stage.unmap();
    Ok(res)
}

/// Context-aware auto dispatcher for Spatial Gating Unit.
pub async fn spatial_gating_auto(
    ctx: Option<&WgpuContext>,
    z1: &[f32],
    z2: &[f32],
    w_sgu: &[f32],
    b_sgu: &[f32],
    batch_size: usize,
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    #[cfg(feature = "wgpu")]
    if let Some(c) = ctx
        && let Ok(res) =
            spatial_gating_wgpu(c, z1, z2, w_sgu, b_sgu, batch_size, seq_len, dim).await
    {
        return res;
    }
    spatial_gating_cpu(z1, z2, w_sgu, b_sgu, batch_size, seq_len, dim)
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
