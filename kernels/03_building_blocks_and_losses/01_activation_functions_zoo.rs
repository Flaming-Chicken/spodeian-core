//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 01:
//! Activation Functions Zoo (Rectified, Bounded, Gated, Periodic, and Self-Normalizing).

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for periodic Snake activation: x + (1 - cos(2*a*x)) / (2*a).
pub const SNAKE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    len: u32,
    a: f32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.len) {
        return;
    }
    let x = in_data[idx];
    let two_a = 2.0 * params.a;
    out_data[idx] = x + (1.0 - cos(two_a * x)) / two_a;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SnakeUniforms {
    len: u32,
    a: f32,
    _pad: [f32; 2],
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

/// Elementwise SiLU: x / (1 + exp(-x))
#[inline]
pub fn silu_val(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Elementwise GELU (erf formulation)
#[inline]
pub fn gelu_val(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

/// High-precision polynomial approximation of error function erf(x).
fn erf_approx(x: f32) -> f32 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x).exp();
    sign * y
}

/// SwiGLU forward on CPU: (silu(x @ w_gate) * (x @ w_up)) @ w_down
pub fn swiglu_cpu(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    batch_size: usize,
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut gated = vec![0.0f32; batch_size * hidden_dim];
    for b in 0..batch_size {
        for h in 0..hidden_dim {
            let mut sum_gate = 0.0f32;
            let mut sum_up = 0.0f32;
            for i in 0..in_dim {
                let x_val = x[b * in_dim + i];
                sum_gate += x_val * w_gate[i * hidden_dim + h];
                sum_up += x_val * w_up[i * hidden_dim + h];
            }
            gated[b * hidden_dim + h] = silu_val(sum_gate) * sum_up;
        }
    }

    let mut out = vec![0.0f32; batch_size * out_dim];
    for b in 0..batch_size {
        for j in 0..out_dim {
            let mut sum = 0.0f32;
            for h in 0..hidden_dim {
                sum += gated[b * hidden_dim + h] * w_down[h * out_dim + j];
            }
            out[b * out_dim + j] = sum;
        }
    }
    out
}

/// GeGLU forward on CPU: (gelu(x @ w_gate) * (x @ w_up)) @ w_down
pub fn geglu_cpu(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    batch_size: usize,
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut gated = vec![0.0f32; batch_size * hidden_dim];
    for b in 0..batch_size {
        for h in 0..hidden_dim {
            let mut sum_gate = 0.0f32;
            let mut sum_up = 0.0f32;
            for i in 0..in_dim {
                let x_val = x[b * in_dim + i];
                sum_gate += x_val * w_gate[i * hidden_dim + h];
                sum_up += x_val * w_up[i * hidden_dim + h];
            }
            gated[b * hidden_dim + h] = gelu_val(sum_gate) * sum_up;
        }
    }

    let mut out = vec![0.0f32; batch_size * out_dim];
    for b in 0..batch_size {
        for j in 0..out_dim {
            let mut sum = 0.0f32;
            for h in 0..hidden_dim {
                sum += gated[b * hidden_dim + h] * w_down[h * out_dim + j];
            }
            out[b * out_dim + j] = sum;
        }
    }
    out
}

/// Periodic Snake activation on CPU: x + (1 - cos(2*a*x)) / (2*a)
pub fn snake_cpu(x: &[f32], a: f32) -> Vec<f32> {
    let two_a = 2.0 * a;
    x.iter()
        .map(|&v| v + (1.0 - (two_a * v).cos()) / two_a)
        .collect()
}

/// Self-normalizing SELU on CPU: where(x > 0, scale * x, scale * alpha * (exp(x) - 1))
pub fn selu_cpu(x: &[f32], scale: f32, alpha: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            if v > 0.0 {
                scale * v
            } else {
                scale * alpha * (v.exp() - 1.0)
            }
        })
        .collect()
}

/// ReLU on CPU
pub fn relu_cpu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// LeakyReLU on CPU
pub fn leaky_relu_cpu(x: &[f32], negative_slope: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| if v > 0.0 { v } else { v * negative_slope })
        .collect()
}

/// SiLU on CPU
pub fn silu_cpu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| silu_val(v)).collect()
}

/// WebGPU accelerated Snake activation.
pub async fn snake_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    in_data: &[f32],
    a: f32,
) -> Option<Vec<f32>> {
    let len = in_data.len() as u32;
    let byte_size = std::mem::size_of_val(in_data) as u64;

    let in_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Snake In Buffer"),
        contents: bytemuck::cast_slice(in_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Snake Out Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniforms = SnakeUniforms {
        len,
        a,
        _pad: [0.0; 2],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Snake Uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Snake Shader"),
        source: wgpu::ShaderSource::Wgsl(SNAKE_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Snake Bind Group Layout"),
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
        label: Some("Snake Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Snake Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Snake Bind Group"),
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
        label: Some("Snake Encoder"),
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Snake Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = len.div_ceil(64);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Snake Staging Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
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

/// Context-aware auto dispatcher for Snake activation.
pub async fn snake_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    x: &[f32],
    a: f32,
) -> Vec<f32> {
    if let Some(ctx) = context
        && let Some(res) = snake_wgpu(&ctx.0, &ctx.1, x, a).await
    {
        return res;
    }
    snake_cpu(x, a)
}
