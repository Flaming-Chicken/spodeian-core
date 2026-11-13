//! Dual-engine WebGPU + CPU Hardware Kernels for Category 01 Topic 13:
//! Information Geometry, Solomonoff MDL, and MaxEnt Free Energy.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for dual alpha-geodesic interpolation.
pub const DUAL_GEODESIC_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> p1: array<f32>;
@group(0) @binding(1) var<storage, read> p2: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_interp: array<f32>;

struct Uniforms {
    dim: u32,
    t: f32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.dim) {
        return;
    }
    out_interp[idx] = (1.0 - params.t) * p1[idx] + params.t * p2[idx];
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GeodesicUniforms {
    dim: u32,
    t: f32,
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

/// Amari dually flat geometry operations on CPU.
pub fn log_partition_cpu(theta: &[f32]) -> f32 {
    let mut max_val = f32::NEG_INFINITY;
    for &val in theta {
        if val > max_val {
            max_val = val;
        }
    }
    let mut sum_exp = 0.0f32;
    for &val in theta {
        sum_exp += (val - max_val).exp();
    }
    max_val + sum_exp.ln()
}

pub fn expectation_param_cpu(theta: &[f32]) -> Vec<f32> {
    let psi = log_partition_cpu(theta);
    theta.iter().map(|&th| (th - psi).exp()).collect()
}

pub fn dual_potential_cpu(eta: &[f32]) -> f32 {
    eta.iter()
        .map(|&e| if e > 0.0 { e * e.ln() } else { 0.0 })
        .sum()
}

pub fn legendre_gap_cpu(theta: &[f32], eta: &[f32]) -> f32 {
    let psi = log_partition_cpu(theta);
    let phi = dual_potential_cpu(eta);
    let inner: f32 = theta.iter().zip(eta.iter()).map(|(&th, &et)| th * et).sum();
    (psi + phi - inner).abs()
}

pub fn fisher_rao_metric_cpu(eta: &[f32]) -> Vec<f32> {
    let dim = eta.len();
    let mut g = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let diag = if i == j { eta[i] } else { 0.0 };
            g[i * dim + j] = diag - eta[i] * eta[j];
        }
    }
    g
}

pub fn e_geodesic_cpu(theta_1: &[f32], theta_2: &[f32], t: f32) -> Vec<f32> {
    theta_1
        .iter()
        .zip(theta_2.iter())
        .map(|(&th1, &th2)| (1.0 - t) * th1 + t * th2)
        .collect()
}

pub fn m_geodesic_eta_cpu(eta_1: &[f32], eta_2: &[f32], t: f32) -> Vec<f32> {
    eta_1
        .iter()
        .zip(eta_2.iter())
        .map(|(&e1, &e2)| (1.0 - t) * e1 + t * e2)
        .collect()
}

pub fn kl_divergence_cpu(eta_p: &[f32], eta_q: &[f32]) -> f32 {
    eta_p
        .iter()
        .zip(eta_q.iter())
        .map(|(&p, &q)| p * (p / q).ln())
        .sum()
}

pub fn bregman_divergence_cpu(theta_1: &[f32], theta_2: &[f32], eta_1: &[f32]) -> f32 {
    let psi_1 = log_partition_cpu(theta_1);
    let psi_2 = log_partition_cpu(theta_2);
    let inner: f32 = eta_1
        .iter()
        .zip(theta_2.iter().zip(theta_1.iter()))
        .map(|(&e, (&th2, &th1))| e * (th2 - th1))
        .sum();
    psi_2 - psi_1 - inner
}

/// Shtarkov NML and Prequential MDL on CPU.
pub fn shtarkov_log_complexity_cpu(n: usize) -> f32 {
    fn comb(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        let mut res = 1.0;
        for i in 1..=k {
            res = res * (n - i + 1) as f64 / i as f64;
        }
        res
    }

    let mut comp_sum = 0.0f64;
    for k in 0..=n {
        let c = comb(n, k);
        let max_lik = if k == 0 || k == n {
            1.0
        } else {
            let p = k as f64 / n as f64;
            p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
        };
        comp_sum += c * max_lik;
    }
    comp_sum.ln() as f32
}

pub fn prequential_codelength_cpu(stream_bits: &[f32]) -> f32 {
    let n_seq = stream_bits.len();
    let mut preq_loss = 0.0f64;
    let mut s_prev = 0.0f64;
    for t in 1..=n_seq {
        let bit = stream_bits[t - 1] as f64;
        let p_pred = (s_prev + 0.5) / (t as f64);
        let prob_t = if bit == 1.0 { p_pred } else { 1.0 - p_pred };
        preq_loss -= prob_t.ln();
        s_prev += bit;
    }
    preq_loss as f32
}

/// Variational Free Energy on CPU.
pub fn compute_variational_free_energy_cpu(
    q_mu: &[f32],
    q_sigma: &[f32],
    p_prior_mu: &[f32],
    p_prior_sigma: &[f32],
    obs_x: &[f32],
    a_mat: &[f32],
    sigma_obs: f32,
) -> (f32, f32, f32) {
    let dim = q_mu.len();
    let mut kl = 0.0f64;
    for i in 0..dim {
        let sq = q_sigma[i] as f64;
        let sp = p_prior_sigma[i] as f64;
        let mq = q_mu[i] as f64;
        let mp = p_prior_mu[i] as f64;
        kl += (sp / sq).ln() + (sq * sq + (mq - mp) * (mq - mp)) / (2.0 * sp * sp) - 0.5;
    }

    let obs_dim = obs_x.len();
    let mut res_mean = vec![0.0f64; obs_dim];
    for i in 0..obs_dim {
        let mut ax = 0.0f64;
        for j in 0..dim {
            ax += (a_mat[i * dim + j] as f64) * (q_mu[j] as f64);
        }
        res_mean[i] = (obs_x[i] as f64) - ax;
    }
    let sq_err_mean: f64 = res_mean.iter().map(|&r| r * r).sum();

    let mut tr_cov = 0.0f64;
    for j in 0..dim {
        let mut col_norm_sq = 0.0f64;
        for i in 0..obs_dim {
            let a_val = a_mat[i * dim + j] as f64;
            col_norm_sq += a_val * a_val;
        }
        let sq = q_sigma[j] as f64;
        tr_cov += col_norm_sq * sq * sq;
    }

    let sig2 = (sigma_obs as f64) * (sigma_obs as f64);
    let c_const = 0.5 * (obs_dim as f64) * (2.0 * std::f64::consts::PI * sig2).ln();
    let recon = c_const + (sq_err_mean + tr_cov) / (2.0 * sig2);
    let fe = kl + recon;

    (kl as f32, recon as f32, fe as f32)
}

/// WebGPU accelerated dual geodesic interpolation.
pub async fn dual_geodesic_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    p1: &[f32],
    p2: &[f32],
    t: f32,
) -> Option<Vec<f32>> {
    let dim = p1.len() as u32;
    let byte_size = std::mem::size_of_val(p1) as u64;

    let p1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("P1 Buffer"),
        contents: bytemuck::cast_slice(p1),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let p2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("P2 Buffer"),
        contents: bytemuck::cast_slice(p2),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Interp Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniforms = GeodesicUniforms {
        dim,
        t,
        _pad: [0.0; 2],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniforms Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dual Geodesic Shader"),
        source: wgpu::ShaderSource::Wgsl(DUAL_GEODESIC_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Geodesic Bind Group Layout"),
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
        label: Some("Geodesic Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Geodesic Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Geodesic Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p1_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: p2_buffer.as_entire_binding(),
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
        label: Some("Geodesic Encoder"),
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Geodesic Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = dim.div_ceil(64);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
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

/// Context-aware auto dispatcher.
pub async fn dual_geodesic_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    p1: &[f32],
    p2: &[f32],
    t: f32,
) -> Vec<f32> {
    if let Some(ctx) = context
        && let Some(res) = dual_geodesic_wgpu(&ctx.0, &ctx.1, p1, p2, t).await
    {
        return res;
    }
    e_geodesic_cpu(p1, p2, t)
}
