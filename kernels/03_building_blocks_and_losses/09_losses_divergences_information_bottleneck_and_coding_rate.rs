//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 09:
//! Losses: Divergences, Information Bottleneck, and Coding Rate.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for elementwise Itakura-Saito distance components.
pub const ITAKURA_SAITO_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_x: array<f32>;
@group(0) @binding(1) var<storage, read> in_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_dist: array<f32>;

struct Uniforms {
    total_elements: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    let xv = max(in_x[idx], params.eps);
    let yv = max(in_y[idx], params.eps);
    let ratio = xv / yv;
    out_dist[idx] = ratio - log(ratio) - 1.0;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ItakuraSaitoUniforms {
    total_elements: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Linear anchor forward on CPU: Y = X @ W + b.
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

/// Forward KL Divergence on CPU.
pub fn kl_forward_cpu(p: &[f32], q: &[f32], b: usize, k: usize, eps: f32) -> f32 {
    let mut total = 0.0f32;
    for i in 0..b {
        for j in 0..k {
            let pv = p[i * k + j].max(eps);
            let qv = q[i * k + j].max(eps);
            total += pv * (pv.ln() - qv.ln());
        }
    }
    total / (b as f32)
}

/// Reverse KL Divergence on CPU.
pub fn kl_reverse_cpu(p: &[f32], q: &[f32], b: usize, k: usize, eps: f32) -> f32 {
    kl_forward_cpu(q, p, b, k, eps)
}

/// Jensen-Shannon Divergence on CPU.
pub fn jsd_cpu(p: &[f32], q: &[f32], b: usize, k: usize) -> f32 {
    let mut m = vec![0.0f32; b * k];
    for i in 0..(b * k) {
        m[i] = 0.5f32 * (p[i] + q[i]);
    }
    0.5f32 * kl_forward_cpu(p, &m, b, k, 1e-12) + 0.5f32 * kl_forward_cpu(q, &m, b, k, 1e-12)
}

/// alpha-Renyi Divergence on CPU.
pub fn renyi_divergence_cpu(p: &[f32], q: &[f32], b: usize, k: usize, alpha: f32, eps: f32) -> f32 {
    let scale = 1.0f32 / (alpha - 1.0f32);
    let mut total = 0.0f32;
    for i in 0..b {
        let mut inner = 0.0f32;
        for j in 0..k {
            let pv = p[i * k + j].max(eps);
            let qv = q[i * k + j].max(eps);
            inner += pv.powf(alpha) * qv.powf(1.0f32 - alpha);
        }
        total += scale * inner.max(eps).ln();
    }
    total / (b as f32)
}

/// Itakura-Saito Distance on CPU.
pub fn itakura_saito_cpu(x: &[f32], y: &[f32], eps: f32) -> f32 {
    let mut total = 0.0f32;
    let n = x.len();
    for i in 0..n {
        let xv = x[i].max(eps);
        let yv = y[i].max(eps);
        let ratio = xv / yv;
        total += ratio - ratio.ln() - 1.0f32;
    }
    total / (n as f32)
}

/// Variational Information Bottleneck (VIB) Gaussian Prior KL on CPU.
pub fn vib_kl_cpu(mu: &[f32], logvar: &[f32], b: usize, d: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..b {
        let mut sum_kl = 0.0f32;
        for j in 0..d {
            let m = mu[i * d + j];
            let s = logvar[i * d + j];
            sum_kl += -0.5f32 * (1.0f32 + s - m * m - s.exp());
        }
        total += sum_kl;
    }
    total / (b as f32)
}

/// Cholesky log-determinant on CPU.
fn cholesky_logdet_cpu(a: &[f32], n: usize) -> f32 {
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                l[i * n + j] = diag.max(1e-12).sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j].max(1e-12);
            }
        }
    }
    let mut logdet = 0.0f32;
    for i in 0..n {
        logdet += 2.0f32 * l[i * n + i].ln();
    }
    logdet
}

/// Maximal Coding Rate Reduction (MCR^2) Total Coding Rate on CPU.
pub fn mcr2_coding_rate_cpu(z: &[f32], n: usize, d: usize, eps: f32) -> f32 {
    let scale = (d as f32) / ((n as f32) * eps.powi(2));
    let mut c = vec![0.0f32; d * d];

    for i in 0..d {
        c[i * d + i] = 1.0f32;
        for j in 0..d {
            let mut dot = 0.0f32;
            for row in 0..n {
                dot += z[row * d + i] * z[row * d + j];
            }
            c[i * d + j] += scale * dot;
        }
    }

    let logdet = cholesky_logdet_cpu(&c, d);
    0.5f32 * logdet
}

/// WebGPU kernel execution for Itakura-Saito distance.
pub async fn itakura_saito_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &[f32],
    y: &[f32],
    eps: f32,
) -> Option<f32> {
    let total_elements = x.len() as u32;
    let byte_size = std::mem::size_of_val(x) as u64;

    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("IS X Buffer"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let y_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("IS Y Buffer"),
        contents: bytemuck::cast_slice(y),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("IS Out Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniforms = ItakuraSaitoUniforms {
        total_elements,
        eps,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("IS Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("IS Staging Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("IS Shader"),
        source: wgpu::ShaderSource::Wgsl(ITAKURA_SAITO_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("IS Layout"),
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
        label: Some("IS Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("IS Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("IS Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y_buffer.as_entire_binding(),
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
        label: Some("IS Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("IS Pass"),
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
    let res_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    let sum: f32 = res_vec.iter().sum();
    Some(sum / (res_vec.len() as f32))
}

/// Context-aware auto dispatcher for Itakura-Saito distance.
pub async fn itakura_saito_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    x: &[f32],
    y: &[f32],
    eps: f32,
) -> f32 {
    if let Some(ctx) = context
        && let Some(res) = itakura_saito_wgpu(&ctx.0, &ctx.1, x, y, eps).await
    {
        return res;
    }
    itakura_saito_cpu(x, y, eps)
}
