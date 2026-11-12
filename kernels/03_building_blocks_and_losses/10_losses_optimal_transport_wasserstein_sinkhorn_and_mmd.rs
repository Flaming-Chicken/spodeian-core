//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 10:
//! Losses: Optimal Transport, Wasserstein, Sinkhorn, and MMD.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// WGSL compute shader for pairwise multi-scale RBF kernel evaluation.
pub const MMD_RBF_KERNEL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_x: array<f32>;
@group(0) @binding(1) var<storage, read> in_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_k: array<f32>;

struct Uniforms {
    n_pts: u32,
    m_pts: u32,
    dim: u32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= params.n_pts || col >= params.m_pts) {
        return;
    }
    var dist_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        let diff = in_x[row * params.dim + d] - in_y[col * params.dim + d];
        dist_sq = dist_sq + diff * diff;
    }
    // Multi-scale RBF: sigmas = 0.5, 1.0, 2.0 -> gamma = 1 / (2 * s^2)
    // s=0.5 -> gamma = 2.0; s=1.0 -> gamma = 0.5; s=2.0 -> gamma = 0.125
    let k_val = exp(-2.0 * dist_sq) + exp(-0.5 * dist_sq) + exp(-0.125 * dist_sq);
    out_k[row * params.m_pts + col] = k_val;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MmdKernelUniforms {
    n_pts: u32,
    m_pts: u32,
    dim: u32,
    _pad: u32,
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

/// Exact 1D Wasserstein Distance on CPU via sorted order statistics.
pub fn wasserstein_1d_cpu(u: &[f32], v: &[f32], b: usize, n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..b {
        let mut u_row = u[i * n..(i + 1) * n].to_vec();
        let mut v_row = v[i * n..(i + 1) * n].to_vec();
        u_row.sort_by(|x, y| x.total_cmp(y));
        v_row.sort_by(|x, y| x.total_cmp(y));
        for j in 0..n {
            total += (u_row[j] - v_row[j]).abs();
        }
    }
    total / ((b * n) as f32)
}

/// Sliced Wasserstein Distance on CPU.
pub fn sliced_wasserstein_cpu(
    x: &[f32],
    y: &[f32],
    projections: &[f32],
    b: usize,
    n: usize,
    d: usize,
    l: usize,
) -> f32 {
    let mut total = 0.0f32;
    for b_idx in 0..b {
        for l_idx in 0..l {
            let proj_slice = &projections[l_idx * d..(l_idx + 1) * d];
            let mut px = vec![0.0f32; n];
            let mut py = vec![0.0f32; n];
            for i in 0..n {
                let x_pt = &x[(b_idx * n + i) * d..(b_idx * n + i + 1) * d];
                let y_pt = &y[(b_idx * n + i) * d..(b_idx * n + i + 1) * d];
                let mut dot_x = 0.0f32;
                let mut dot_y = 0.0f32;
                for k in 0..d {
                    dot_x += x_pt[k] * proj_slice[k];
                    dot_y += y_pt[k] * proj_slice[k];
                }
                px[i] = dot_x;
                py[i] = dot_y;
            }
            px.sort_by(|p1, p2| p1.total_cmp(p2));
            py.sort_by(|p1, p2| p1.total_cmp(p2));
            for i in 0..n {
                total += (px[i] - py[i]).abs();
            }
        }
    }
    total / ((b * l * n) as f32)
}

/// Pairwise squared Euclidean distances on CPU.
fn pairwise_sqdist_cpu(x: &[f32], y: &[f32], n: usize, m: usize, d: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum_sq = 0.0f32;
            for k in 0..d {
                let diff = x[i * d + k] - y[j * d + k];
                sum_sq += diff * diff;
            }
            c[i * m + j] = sum_sq;
        }
    }
    c
}

/// Entropic Regularized Optimal Transport (Sinkhorn) on CPU.
pub fn sinkhorn_cpu(
    x: &[f32],
    y: &[f32],
    n: usize,
    m: usize,
    d: usize,
    eps: f32,
    num_iters: usize,
) -> f32 {
    let c = pairwise_sqdist_cpu(x, y, n, m, d);
    let mut k = vec![0.0f32; n * m];
    for i in 0..n * m {
        k[i] = (-c[i] / eps).exp();
    }

    let a_val = 1.0f32 / (n as f32);
    let b_val = 1.0f32 / (m as f32);
    let mut u = vec![a_val; n];
    let mut v = vec![b_val; m];

    for _ in 0..num_iters {
        for i in 0..n {
            let mut kv_i = 0.0f32;
            for j in 0..m {
                kv_i += k[i * m + j] * v[j];
            }
            u[i] = a_val / (kv_i + 1e-12);
        }
        for j in 0..m {
            let mut ktu_j = 0.0f32;
            for i in 0..n {
                ktu_j += k[i * m + j] * u[i];
            }
            v[j] = b_val / (ktu_j + 1e-12);
        }
    }

    let mut cost = 0.0f32;
    for i in 0..n {
        for j in 0..m {
            let p_ij = u[i] * k[i * m + j] * v[j];
            cost += p_ij * c[i * m + j];
        }
    }
    cost
}

/// Debiased Sinkhorn Divergence on CPU.
pub fn debiased_sinkhorn_cpu(
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    eps: f32,
    num_iters: usize,
) -> f32 {
    let s_xy = sinkhorn_cpu(x, y, n, n, d, eps, num_iters);
    let s_xx = sinkhorn_cpu(x, x, n, n, d, eps, num_iters);
    let s_yy = sinkhorn_cpu(y, y, n, n, d, eps, num_iters);
    s_xy - 0.5f32 * s_xx - 0.5f32 * s_yy
}

/// Maximum Mean Discrepancy (MMD^2) on CPU.
pub fn mmd_cpu(x: &[f32], y: &[f32], n: usize, m: usize, d: usize, sigmas: &[f32]) -> f32 {
    let eval_k_mean = |a: &[f32], b: &[f32], n_a: usize, n_b: usize| -> f32 {
        let dists = pairwise_sqdist_cpu(a, b, n_a, n_b, d);
        let mut sum_k = 0.0f32;
        for &d2 in &dists {
            for &s in sigmas {
                let gamma = 1.0f32 / (2.0f32 * s * s);
                sum_k += (-gamma * d2).exp();
            }
        }
        sum_k / ((n_a * n_b) as f32)
    };

    let mean_xx = eval_k_mean(x, x, n, n);
    let mean_yy = eval_k_mean(y, y, m, m);
    let mean_xy = eval_k_mean(x, y, n, m);

    mean_xx + mean_yy - 2.0f32 * mean_xy
}

/// Energy Distance on CPU.
pub fn energy_distance_cpu(x: &[f32], y: &[f32], n: usize, m: usize, d: usize) -> f32 {
    let eval_dist_mean = |a: &[f32], b: &[f32], n_a: usize, n_b: usize| -> f32 {
        let mut sum_dist = 0.0f32;
        for i in 0..n_a {
            for j in 0..n_b {
                let mut sum_sq = 0.0f32;
                for k in 0..d {
                    let diff = a[i * d + k] - b[j * d + k];
                    sum_sq += diff * diff;
                }
                sum_dist += sum_sq.sqrt();
            }
        }
        sum_dist / ((n_a * n_b) as f32)
    };

    let mean_xy = eval_dist_mean(x, y, n, m);
    let mean_xx = eval_dist_mean(x, x, n, n);
    let mean_yy = eval_dist_mean(y, y, m, m);

    2.0f32 * mean_xy - mean_xx - mean_yy
}

/// Computes pairwise RBF kernel matrix via WebGPU WGSL shader.
pub fn mmd_kernel_wgpu(
    context: &crate::WgpuContext,
    x: &[f32],
    y: &[f32],
    n: usize,
    m: usize,
    d: usize,
) -> Option<Vec<f32>> {
    let device = &context.device;
    let queue = &context.queue;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mmd_rbf_kernel_shader"),
        source: wgpu::ShaderSource::Wgsl(MMD_RBF_KERNEL_SHADER.into()),
    });

    let buffer_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buffer_x"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_y = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buffer_y"),
        contents: bytemuck::cast_slice(y),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = (n * m * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer_out"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = MmdKernelUniforms {
        n_pts: n as u32,
        m_pts: m as u32,
        dim: d as u32,
        _pad: 0,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniform_buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mmd_bind_group_layout"),
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mmd_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mmd_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mmd_compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mmd_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mmd_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = ((n as u32) + 7) / 8;
        let workgroups_y = ((m as u32) + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    encoder.copy_buffer_to_buffer(&buffer_out, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device.poll(wgpu::Maintain::Wait);

    if let Ok(Ok(())) = receiver.recv() {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Some(result)
    } else {
        None
    }
}

/// Auto-dispatcher: evaluates MMD using WebGPU WGSL if context available, else pure-CPU.
pub fn mmd_auto(
    context: Option<&crate::WgpuContext>,
    x: &[f32],
    y: &[f32],
    n: usize,
    m: usize,
    d: usize,
    sigmas: &[f32],
) -> f32 {
    if let Some(ctx) = context {
        if let (Some(k_xx), Some(k_yy), Some(k_xy)) = (
            mmd_kernel_wgpu(ctx, x, x, n, n, d),
            mmd_kernel_wgpu(ctx, y, y, m, m, d),
            mmd_kernel_wgpu(ctx, x, y, n, m, d),
        ) {
            let mean_xx = k_xx.iter().sum::<f32>() / (k_xx.len() as f32);
            let mean_yy = k_yy.iter().sum::<f32>() / (k_yy.len() as f32);
            let mean_xy = k_xy.iter().sum::<f32>() / (k_xy.len() as f32);
            return mean_xx + mean_yy - 2.0f32 * mean_xy;
        }
    }
    mmd_cpu(x, y, n, m, d, sigmas)
}
