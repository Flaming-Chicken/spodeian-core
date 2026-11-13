//! Dual-engine WebGPU + CPU Hardware Kernels for Category 01 Topic 12:
//! Inductive vs. Transductive Learning and Online Regret vs. Batch IID.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for FTRL-Proximal coordinate thresholding.
pub const FTRL_THRESHOLD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> z_accum: array<f32>;
@group(0) @binding(1) var<storage, read> n_accum: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_w: array<f32>;

struct Uniforms {
    dim: u32,
    alpha: f32,
    beta: f32,
    lambda1: f32,
    lambda2: f32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.dim) {
        return;
    }
    let z = z_accum[idx];
    let n = n_accum[idx];
    if (abs(z) <= params.lambda1) {
        out_w[idx] = 0.0;
    } else {
        let sign_z = select(-1.0, 1.0, z > 0.0);
        let denom = (params.beta + sqrt(n)) / params.alpha + params.lambda2;
        out_w[idx] = - (z - sign_z * params.lambda1) / denom;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FtrlUniforms {
    dim: u32,
    alpha: f32,
    beta: f32,
    lambda1: f32,
    lambda2: f32,
    _pad: [f32; 3],
}

/// Helper matrix inversion via Gauss-Jordan elimination on CPU.
fn invert_matrix_cpu(mat: &[f32], n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            a[i * 2 * n + j] = mat[i * n + j];
        }
        a[i * 2 * n + n + i] = 1.0;
    }

    for i in 0..n {
        let mut pivot_row = i;
        let mut max_val = a[i * 2 * n + i].abs();
        for r in (i + 1)..n {
            let val = a[r * 2 * n + i].abs();
            if val > max_val {
                max_val = val;
                pivot_row = r;
            }
        }
        if pivot_row != i {
            for c in 0..(2 * n) {
                a.swap(i * 2 * n + c, pivot_row * 2 * n + c);
            }
        }

        let inv_p = 1.0 / a[i * 2 * n + i];
        for c in 0..(2 * n) {
            a[i * 2 * n + c] *= inv_p;
        }
        for r in 0..n {
            if r != i {
                let factor = a[r * 2 * n + i];
                for c in 0..(2 * n) {
                    a[r * 2 * n + c] -= factor * a[i * 2 * n + c];
                }
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i * 2 * n + n + j];
        }
    }
    inv
}

/// Transductive label spreading on CPU.
pub fn transductive_label_spreading_cpu(
    laplacian_s: &[f32],
    y_init: &[f32],
    num_nodes: usize,
    num_classes: usize,
    alpha: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut m = vec![0.0f32; num_nodes * num_nodes];
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            let eye = if i == j { 1.0f32 } else { 0.0f32 };
            m[i * num_nodes + j] = eye - alpha * laplacian_s[i * num_nodes + j];
        }
    }

    let inv_m = invert_matrix_cpu(&m, num_nodes);

    let mut f_star = vec![0.0f32; num_nodes * num_classes];
    for i in 0..num_nodes {
        for c in 0..num_classes {
            let mut sum_val = 0.0f32;
            for j in 0..num_nodes {
                sum_val += inv_m[i * num_nodes + j] * y_init[j * num_classes + c];
            }
            f_star[i * num_classes + c] = sum_val;
        }
    }

    let mut f_probs = vec![0.0f32; num_nodes * num_classes];
    let mut preds = vec![0.0f32; num_nodes];

    for i in 0..num_nodes {
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let val = f_star[i * num_classes + c];
            if val > max_val {
                max_val = val;
            }
        }
        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            let exp_val = (f_star[i * num_classes + c] - max_val).exp();
            f_probs[i * num_classes + c] = exp_val;
            sum_exp += exp_val;
        }
        let mut best_c = 0usize;
        let mut best_p = -1.0f32;
        for c in 0..num_classes {
            f_probs[i * num_classes + c] /= sum_exp;
            if f_probs[i * num_classes + c] > best_p {
                best_p = f_probs[i * num_classes + c];
                best_c = c;
            }
        }
        preds[i] = best_c as f32;
    }

    (f_star, f_probs, preds)
}

/// Inductive linear predictor on CPU.
pub fn inductive_predictor_cpu(
    x_test: &[f32],
    w_ind: &[f32],
    batch: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut scores = vec![0.0f32; batch];
    let mut preds = vec![0.0f32; batch];
    for b in 0..batch {
        let mut s = 0.0f32;
        for d in 0..dim {
            s += x_test[b * dim + d] * w_ind[d];
        }
        scores[b] = s;
        preds[b] = if s >= 0.5f32 { 1.0f32 } else { 0.0f32 };
    }
    (scores, preds)
}

/// FTRL coordinate threshold evaluation on CPU.
pub fn ftrl_threshold_cpu(
    z: &[f32],
    n: &[f32],
    dim: usize,
    alpha: f32,
    beta: f32,
    lambda1: f32,
    lambda2: f32,
) -> Vec<f32> {
    let mut w = vec![0.0f32; dim];
    for i in 0..dim {
        if z[i].abs() <= lambda1 {
            w[i] = 0.0;
        } else {
            let sign_z = if z[i] > 0.0 { 1.0f32 } else { -1.0f32 };
            let denom = (beta + n[i].sqrt()) / alpha + lambda2;
            w[i] = -(z[i] - sign_z * lambda1) / denom;
        }
    }
    w
}

/// FTRL-Proximal online optimizer full update sequence on CPU.
pub fn ftrl_proximal_cpu(
    grad_stream: &[f32],
    time_steps: usize,
    dim: usize,
    alpha: f32,
    beta: f32,
    lambda1: f32,
    lambda2: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut z = vec![0.0f32; dim];
    let mut n = vec![0.0f32; dim];

    for t in 0..time_steps {
        let mut w_t = vec![0.0f32; dim];
        for i in 0..dim {
            if z[i].abs() <= lambda1 {
                w_t[i] = 0.0;
            } else {
                let sign_z = if z[i] > 0.0 { 1.0f32 } else { -1.0f32 };
                let denom = (beta + n[i].sqrt()) / alpha + lambda2;
                w_t[i] = -(z[i] - sign_z * lambda1) / denom;
            }
        }

        for i in 0..dim {
            let g = grad_stream[t * dim + i];
            let sigma_i = ((n[i] + g * g).sqrt() - n[i].sqrt()) / alpha;
            z[i] += g - sigma_i * w_t[i];
            n[i] += g * g;
        }
    }

    let final_w = ftrl_threshold_cpu(&z, &n, dim, alpha, beta, lambda1, lambda2);
    (final_w, z, n)
}

/// Page-Hinkley CUSUM concept drift detection on CPU.
pub fn page_hinkley_cusum_cpu(
    stream_errors: &[f32],
    delta: f32,
    threshold: f32,
    mean_est: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = stream_errors.len();
    let mut cusum_s = Vec::with_capacity(len);
    let mut ph_stat = Vec::with_capacity(len);
    let mut drift_flags = Vec::with_capacity(len);

    let mut cum_s = 0.0f32;
    let mut min_s = 0.0f32;

    for &e in stream_errors {
        cum_s += e - mean_est - delta;
        if cum_s < min_s {
            min_s = cum_s;
        }
        let stat = cum_s - min_s;
        cusum_s.push(cum_s);
        ph_stat.push(stat);
        drift_flags.push(if stat > threshold { 1.0f32 } else { 0.0f32 });
    }

    (cusum_s, ph_stat, drift_flags)
}

/// Dual-engine automatic execution for FTRL coordinate thresholding.
pub fn ftrl_threshold_auto(
    device: Option<&Arc<wgpu::Device>>,
    queue: Option<&Arc<wgpu::Queue>>,
    z: &[f32],
    n: &[f32],
    dim: usize,
    alpha: f32,
    beta: f32,
    lambda1: f32,
    lambda2: f32,
) -> Vec<f32> {
    if let (Some(d), Some(q)) = (device, queue)
        && let Some(res) = ftrl_threshold_wgpu(d, q, z, n, dim, alpha, beta, lambda1, lambda2)
    {
        return res;
    }
    ftrl_threshold_cpu(z, n, dim, alpha, beta, lambda1, lambda2)
}

/// WebGPU compute pipeline execution for FTRL coordinate thresholding.
pub fn ftrl_threshold_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    z: &[f32],
    n: &[f32],
    dim: usize,
    alpha: f32,
    beta: f32,
    lambda1: f32,
    lambda2: f32,
) -> Option<Vec<f32>> {
    if dim == 0 {
        return None;
    }

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ftrl_threshold_shader"),
        source: wgpu::ShaderSource::Wgsl(FTRL_THRESHOLD_SHADER.into()),
    });

    let bytes_len = (dim * std::mem::size_of::<f32>()) as u64;

    let z_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("z_buf"),
        contents: bytemuck::cast_slice(z),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("n_buf"),
        contents: bytemuck::cast_slice(n),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_w_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_w_buf"),
        size: bytes_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buf"),
        size: bytes_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = FtrlUniforms {
        dim: dim as u32,
        alpha,
        beta,
        lambda1,
        lambda2,
        _pad: [0.0; 3],
    };
    let u_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ftrl_uniforms_buf"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ftrl_bgl"),
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
        label: Some("ftrl_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ftrl_pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ftrl_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: z_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: n_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: u_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ftrl_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ftrl_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (dim as u32).div_ceil(64);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&out_w_buf, 0, &staging_buf, 0, bytes_len);
    queue.submit(Some(encoder.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });

    let _ = device.poll(wgpu::Maintain::Wait);

    if rx.recv().ok()?.is_ok() {
        let data = slice.get_mapped_range();
        let w: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();
        Some(w)
    } else {
        None
    }
}
