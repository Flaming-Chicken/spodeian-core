//! Dual-engine WebGPU + CPU Hardware Kernels for Category 01 Topic 11:
//! Statistical Inference: Bayesian, Frequentist, Likelihoodist, MDL, and PAC-Bayes.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for James-Stein shrinkage estimator.
pub const JAMES_STEIN_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> y_obs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_theta_js: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_theta_js_pos: array<f32>;

struct Uniforms {
    num_groups: u32,
    sigma2: f32,
    c_shrink: f32,
    c_shrink_pos: f32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_groups) {
        return;
    }
    let y = y_obs[idx];
    out_theta_js[idx] = params.c_shrink * y;
    out_theta_js_pos[idx] = params.c_shrink_pos * y;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct JamesSteinUniforms {
    num_groups: u32,
    sigma2: f32,
    c_shrink: f32,
    c_shrink_pos: f32,
}

/// Frequentist Gaussian mean MLE, Fisher information, and Cramér-Rao lower bound on CPU.
pub fn frequentist_fisher_cpu(
    x_data: &[f32],
    num_obs: usize,
    obs_dim: usize,
    sigma2_noise: f32,
    theta_0: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let mut theta_mle = vec![0.0f32; obs_dim];
    for i in 0..num_obs {
        for d in 0..obs_dim {
            theta_mle[d] += x_data[i * obs_dim + d];
        }
    }
    for d in 0..obs_dim {
        theta_mle[d] /= num_obs as f32;
    }

    let fisher_val = (num_obs as f32) / sigma2_noise;
    let crlb_val = sigma2_noise / (num_obs as f32);
    let fisher_diag = vec![fisher_val; obs_dim];
    let cramer_rao = vec![crlb_val; obs_dim];

    let mut diff_sq_sum = 0.0f32;
    for d in 0..obs_dim {
        let diff = theta_mle[d] - theta_0[d];
        diff_sq_sum += diff * diff;
    }
    let lrt_stat = (num_obs as f32 / sigma2_noise) * diff_sq_sum;

    (theta_mle, fisher_diag, cramer_rao, lrt_stat)
}

/// Bayesian conjugate Gaussian update and Laplace marginal likelihood Occam factor on CPU.
pub fn bayesian_conjugate_laplace_cpu(
    x_data: &[f32],
    mu_0: &[f32],
    num_obs: usize,
    obs_dim: usize,
    sigma2_0: f32,
    sigma2_noise: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32) {
    let mut theta_mle = vec![0.0f32; obs_dim];
    for i in 0..num_obs {
        for d in 0..obs_dim {
            theta_mle[d] += x_data[i * obs_dim + d];
        }
    }
    for d in 0..obs_dim {
        theta_mle[d] /= num_obs as f32;
    }

    let prec_0 = 1.0f32 / sigma2_0;
    let prec_lik = (num_obs as f32) / sigma2_noise;
    let prec_post = prec_0 + prec_lik;
    let var_post = 1.0f32 / prec_post;
    let sigma_post = var_post.sqrt();

    let mut mu_post = vec![0.0f32; obs_dim];
    for d in 0..obs_dim {
        mu_post[d] = var_post * (prec_0 * mu_0[d] + prec_lik * theta_mle[d]);
    }
    let sigma_post_vec = vec![sigma_post; obs_dim];

    let log_det_h = (obs_dim as f32) * prec_post.ln();
    let mut res_sq_sum = 0.0f32;
    for i in 0..num_obs {
        for d in 0..obs_dim {
            let res = x_data[i * obs_dim + d] - mu_post[d];
            res_sq_sum += res * res;
        }
    }

    let pi = std::f32::consts::PI;
    let log_lik_map =
        -0.5f32 * (num_obs as f32) * (obs_dim as f32) * (2.0f32 * pi * sigma2_noise).ln()
            - 0.5f32 * res_sq_sum / sigma2_noise;

    let mut prior_sq_sum = 0.0f32;
    for d in 0..obs_dim {
        let diff = mu_post[d] - mu_0[d];
        prior_sq_sum += diff * diff;
    }
    let log_prior_map = -0.5f32 * (obs_dim as f32) * (2.0f32 * pi * sigma2_0).ln()
        - 0.5f32 * prior_sq_sum / sigma2_0;

    let occam_penalty = -0.5f32 * log_det_h + 0.5f32 * (obs_dim as f32) * (2.0f32 * pi).ln();
    let laplace_log_marginal = log_lik_map + log_prior_map + occam_penalty;

    (mu_post, sigma_post_vec, laplace_log_marginal, occam_penalty)
}

/// James-Stein shrinkage estimator on CPU.
pub fn james_stein_cpu(y_obs: &[f32], sigma2: f32) -> (Vec<f32>, Vec<f32>, f32) {
    let k = y_obs.len();
    let mut norm_sq = 0.0f32;
    for &y in y_obs {
        norm_sq += y * y;
    }
    let c_shrink = 1.0f32 - ((k - 2) as f32 * sigma2) / norm_sq;
    let c_shrink_pos = c_shrink.max(0.0f32);

    let mut theta_js = vec![0.0f32; k];
    let mut theta_js_pos = vec![0.0f32; k];
    for i in 0..k {
        theta_js[i] = c_shrink * y_obs[i];
        theta_js_pos[i] = c_shrink_pos * y_obs[i];
    }
    (theta_js, theta_js_pos, c_shrink)
}

/// McAllester PAC-Bayes generalization bound on CPU.
pub fn pac_bayes_cpu(
    mu_q: &[f32],
    sigma_q: &[f32],
    emp_risk: f32,
    num_obs: usize,
    sigma_p: f32,
    delta: f32,
) -> (f32, f32) {
    let dim = mu_q.len();
    let mut kl_sum = 0.0f32;
    let p_sq = sigma_p * sigma_p;
    for d in 0..dim {
        let m = mu_q[d];
        let s = sigma_q[d];
        let s_sq = s * s;
        kl_sum += (m * m + s_sq) / p_sq - 1.0f32 - (s_sq / p_sq).ln();
    }
    let kl_div = 0.5f32 * kl_sum;
    let complexity_term = ((kl_div + (2.0f32 * (num_obs as f32).sqrt() / delta).ln())
        / (2.0f32 * num_obs as f32))
        .sqrt();
    let pac_bound = emp_risk + complexity_term;
    (kl_div, pac_bound)
}

/// Anytime-valid test martingale with Ville's inequality threshold on CPU.
pub fn anytime_valid_martingale_cpu(e_values: &[f32], alpha: f32) -> (Vec<f32>, f32) {
    let mut prod = 1.0f32;
    let mut martingale = Vec::with_capacity(e_values.len());
    for &ev in e_values {
        prod *= ev;
        martingale.push(prod);
    }
    let threshold = 1.0f32 / alpha;
    (martingale, threshold)
}

/// Automatic dual-engine James-Stein execution with WebGPU fallback to CPU.
pub fn james_stein_auto(
    device: Option<&Arc<wgpu::Device>>,
    queue: Option<&Arc<wgpu::Queue>>,
    y_obs: &[f32],
    sigma2: f32,
) -> (Vec<f32>, Vec<f32>, f32) {
    if let (Some(d), Some(q)) = (device, queue) {
        if let Some(res) = james_stein_wgpu(d, q, y_obs, sigma2) {
            return res;
        }
    }
    james_stein_cpu(y_obs, sigma2)
}

/// WebGPU compute pipeline execution for James-Stein shrinkage.
pub fn james_stein_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    y_obs: &[f32],
    sigma2: f32,
) -> Option<(Vec<f32>, Vec<f32>, f32)> {
    let num_groups = y_obs.len();
    if num_groups == 0 {
        return None;
    }

    let mut norm_sq = 0.0f32;
    for &y in y_obs {
        norm_sq += y * y;
    }
    let c_shrink = 1.0f32 - ((num_groups - 2) as f32 * sigma2) / norm_sq;
    let c_shrink_pos = c_shrink.max(0.0f32);

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("james_stein_shader"),
        source: wgpu::ShaderSource::Wgsl(JAMES_STEIN_SHADER.into()),
    });

    let bytes_len = (num_groups * std::mem::size_of::<f32>()) as u64;

    let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("y_obs_buf"),
        contents: bytemuck::cast_slice(y_obs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_js_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_js_buf"),
        size: bytes_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let out_js_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_js_pos_buf"),
        size: bytes_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_js = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_js"),
        size: bytes_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_pos = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_pos"),
        size: bytes_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = JamesSteinUniforms {
        num_groups: num_groups as u32,
        sigma2,
        c_shrink,
        c_shrink_pos,
    };
    let u_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms_buf"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("js_bgl"),
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
        label: Some("js_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("js_pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("js_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_js_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_js_pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: u_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("js_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("js_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((num_groups as u32) + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&out_js_buf, 0, &staging_js, 0, bytes_len);
    encoder.copy_buffer_to_buffer(&out_js_pos_buf, 0, &staging_pos, 0, bytes_len);
    queue.submit(Some(encoder.finish()));

    let slice_js = staging_js.slice(..);
    let (tx1, rx1) = std::sync::mpsc::channel();
    slice_js.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx1.send(res);
    });

    let slice_pos = staging_pos.slice(..);
    let (tx2, rx2) = std::sync::mpsc::channel();
    slice_pos.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx2.send(res);
    });

    let _ = device.poll(wgpu::Maintain::Wait);

    if rx1.recv().ok()?.is_ok() && rx2.recv().ok()?.is_ok() {
        let data1 = slice_js.get_mapped_range();
        let theta_js: Vec<f32> = bytemuck::cast_slice(&data1).to_vec();
        drop(data1);
        staging_js.unmap();

        let data2 = slice_pos.get_mapped_range();
        let theta_js_pos: Vec<f32> = bytemuck::cast_slice(&data2).to_vec();
        drop(data2);
        staging_pos.unmap();

        Some((theta_js, theta_js_pos, c_shrink))
    } else {
        None
    }
}
