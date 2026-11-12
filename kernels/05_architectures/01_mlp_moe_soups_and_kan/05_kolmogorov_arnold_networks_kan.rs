//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Topic 05:
//! Kolmogorov-Arnold Networks (Chebyshev Polynomial Basis Computation).

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for Chebyshev Polynomial Basis Evaluation:
/// u = tanh(x)
/// T_0 = 1.0, T_1 = u, T_d = 2.0 * u * T_{d-1} - T_{d-2}
pub const CHEBYSHEV_BASIS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_basis: array<f32>;

struct Uniforms {
    num_elements: u32,
    degree: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }
    let x_val = input_x[idx];
    let u = tanh(x_val);
    let deg_plus_1 = params.degree + 1u;
    let base_idx = idx * deg_plus_1;

    output_basis[base_idx + 0u] = 1.0;
    if (params.degree >= 1u) {
        output_basis[base_idx + 1u] = u;
    }

    var t_prev2: f32 = 1.0;
    var t_prev1: f32 = u;
    for (var d: u32 = 2u; d <= params.degree; d = d + 1u) {
        let t_curr = 2.0 * u * t_prev1 - t_prev2;
        output_basis[base_idx + d] = t_curr;
        t_prev2 = t_prev1;
        t_prev1 = t_curr;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChebyBasisUniforms {
    num_elements: u32,
    degree: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Exact CPU reference implementation of Chebyshev basis computation.
pub fn chebyshev_basis_cpu(x: &[f32], degree: usize) -> Vec<f32> {
    let num_elements = x.len();
    let deg_plus_1 = degree + 1;
    let mut out = vec![0.0f32; num_elements * deg_plus_1];

    for (idx, &x_val) in x.iter().enumerate() {
        let u = x_val.tanh();
        let base_idx = idx * deg_plus_1;

        out[base_idx] = 1.0;
        if degree >= 1 {
            out[base_idx + 1] = u;
        }

        let mut t_prev2 = 1.0f32;
        let mut t_prev1 = u;
        for d in 2..=degree {
            let t_curr = 2.0 * u * t_prev1 - t_prev2;
            out[base_idx + d] = t_curr;
            t_prev2 = t_prev1;
            t_prev1 = t_curr;
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

/// WGPU compute implementation for Chebyshev basis computation.
pub fn chebyshev_basis_wgpu(ctx: &WgpuContext, x: &[f32], degree: usize) -> Option<Vec<f32>> {
    use wgpu::util::DeviceExt;

    let num_elements = x.len();
    let deg_plus_1 = degree + 1;
    let total_elements = num_elements * deg_plus_1;
    let byte_size = (total_elements * std::mem::size_of::<f32>()) as u64;

    let in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cheby_in_buf"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheby_out_buf"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cheby_readback_buf"),
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let uniforms = ChebyBasisUniforms {
        num_elements: num_elements as u32,
        degree: degree as u32,
        _pad1: 0,
        _pad2: 0,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cheby_uniforms_buf"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cheby_basis_shader"),
            source: wgpu::ShaderSource::Wgsl(CHEBYSHEV_BASIS_SHADER.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cheby_basis_bgl"),
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

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cheby_basis_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cheby_basis_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cheby_basis_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: u_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cheby_basis_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cheby_basis_pass"),
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

/// Automatically dispatches Chebyshev basis computation: runs WGPU if available, falls back to CPU.
pub fn chebyshev_basis_auto(ctx: Option<&WgpuContext>, x: &[f32], degree: usize) -> Vec<f32> {
    if let Some(c) = ctx {
        if let Some(gpu_out) = chebyshev_basis_wgpu(c, x, degree) {
            return gpu_out;
        }
    }
    chebyshev_basis_cpu(x, degree)
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
