//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 06:
//! 06_naive_bayes_bayesian_regression_and_gaussian_processes.

use crate::context::KernelDevice;

/// WGSL Compute Shader for pairwise RBF Kernel: K[i, j] = sigma_f^2 * exp(-0.5 * ||X_i - X_j||^2 / l^2).
pub const RBF_KERNEL_SHADER: &str = r#"
struct KernelUniforms {
    n: u32,
    dim: u32,
    lengthscale: f32,
    sigma_f: f32,
};

@group(0) @binding(0) var<uniform> uniforms: KernelUniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= uniforms.n || col >= uniforms.n) {
        return;
    }

    var dist_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < uniforms.dim; d = d + 1u) {
        let diff = x[row * uniforms.dim + d] - x[col * uniforms.dim + d];
        dist_sq = dist_sq + diff * diff;
    }

    let inv_two_l_sq = -0.5 / (uniforms.lengthscale * uniforms.lengthscale);
    let val = uniforms.sigma_f * uniforms.sigma_f * exp(dist_sq * inv_two_l_sq);
    k_out[row * uniforms.n + col] = val;
}
"#;

/// CPU reference calculation of RBF covariance matrix.
pub fn rbf_kernel_cpu(x: &[f32], n: usize, dim: usize, lengthscale: f32, sigma_f: f32) -> Vec<f32> {
    let mut k = vec![0.0f32; n * n];
    let inv_two_l_sq = -0.5f32 / (lengthscale * lengthscale);
    let sf2 = sigma_f * sigma_f;

    for i in 0..n {
        for j in 0..n {
            let mut dist_sq = 0.0f32;
            for d in 0..dim {
                let diff = x[i * dim + d] - x[j * dim + d];
                dist_sq += diff * diff;
            }
            k[i * n + j] = sf2 * (dist_sq * inv_two_l_sq).exp();
        }
    }
    k
}

/// Dispatches RBF Kernel compute shader on WebGPU hardware or falls back to CPU.
pub fn rbf_kernel_auto(
    device: Option<&KernelDevice>,
    x: &[f32],
    n: usize,
    dim: usize,
    lengthscale: f32,
    sigma_f: f32,
) -> Vec<f32> {
    let dev = match device {
        Some(d) => d,
        None => return rbf_kernel_cpu(x, n, dim, lengthscale, sigma_f),
    };

    match dev {
        KernelDevice::Cpu => rbf_kernel_cpu(x, n, dim, lengthscale, sigma_f),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            let total_elements = n * n;
            let out_size = (total_elements * std::mem::size_of::<f32>()) as u64;

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct KernelUniforms {
                n: u32,
                dim: u32,
                lengthscale: f32,
                sigma_f: f32,
            }

            let uniforms = KernelUniforms {
                n: n as u32,
                dim: dim as u32,
                lengthscale,
                sigma_f,
            };

            let uniform_buffer =
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RBF Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let x_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RBF X Buffer"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RBF Out Storage Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let readback_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RBF Readback Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RBF Kernel Shader"),
                source: wgpu::ShaderSource::Wgsl(RBF_KERNEL_SHADER.into()),
            });

            let bind_group_layout =
                wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RBF Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
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
                    ],
                });

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RBF Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout =
                wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RBF Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RBF Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RBF Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RBF Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let wg_x = ((n + 15) / 16) as u32;
                let wg_y = ((n + 15) / 16) as u32;
                compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            encoder.copy_buffer_to_buffer(&out_buffer, 0, &readback_buffer, 0, out_size);
            queue.submit(Some(encoder.finish()));

            let buffer_slice = readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
                tx.send(res).unwrap();
            });

            wgpu_device.poll(wgpu::Maintain::Wait);
            if rx.recv().is_err() {
                return rbf_kernel_cpu(x, n, dim, lengthscale, sigma_f);
            }

            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            readback_buffer.unmap();

            result
        }
        #[cfg(not(feature = "wgpu"))]
        KernelDevice::Wgpu(_) => rbf_kernel_cpu(x, n, dim, lengthscale, sigma_f),
    }
}

/// Gaussian Naive Bayes Log-Probabilities on CPU.
pub fn gaussian_naive_bayes_log_probs_cpu(
    x: &[f32],
    n: usize,
    dim: usize,
    means: &[f32],
    vars: &[f32],
    log_priors: &[f32],
    n_classes: usize,
) -> Vec<f32> {
    let mut log_joint = vec![0.0f32; n * n_classes];
    let two_pi = 2.0f32 * std::f32::consts::PI;

    for i in 0..n {
        let mut max_log = f32::NEG_INFINITY;
        for c in 0..n_classes {
            let mut sum_ll = log_priors[c];
            for d in 0..dim {
                let mean_cd = means[c * dim + d];
                let var_cd = vars[c * dim + d];
                let diff = x[i * dim + d] - mean_cd;
                sum_ll += -0.5f32 * ((two_pi * var_cd).ln() + (diff * diff) / var_cd);
            }
            log_joint[i * n_classes + c] = sum_ll;
            if sum_ll > max_log {
                max_log = sum_ll;
            }
        }

        // Log-sum-exp normalization
        let mut sum_exp = 0.0f32;
        for c in 0..n_classes {
            sum_exp += (log_joint[i * n_classes + c] - max_log).exp();
        }
        let lse = max_log + sum_exp.ln();
        for c in 0..n_classes {
            log_joint[i * n_classes + c] -= lse;
        }
    }

    log_joint
}

/// Conjugate Bayesian Linear Regression predictive mean and variance on CPU.
pub fn bayesian_linear_regression_predict_cpu(
    x_star: &[f32],
    n: usize,
    dim: usize,
    m_n: &[f32],
    s_n: &[f32],
    beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f32; n];
    let mut variance = vec![0.0f32; n];
    let inv_beta = 1.0f32 / beta;

    for i in 0..n {
        let mut m_val = 0.0f32;
        for d in 0..dim {
            m_val += x_star[i * dim + d] * m_n[d];
        }
        mean[i] = m_val;

        // var = 1/beta + x_*^T S_N x_*
        let mut quad = 0.0f32;
        for j in 0..dim {
            let mut s_row = 0.0f32;
            for k in 0..dim {
                s_row += x_star[i * dim + k] * s_n[k * dim + j];
            }
            quad += s_row * x_star[i * dim + j];
        }
        variance[i] = inv_beta + quad;
    }

    (mean, variance)
}

/// Exact Gaussian Process predictive mean on CPU: mu_* = K_* alpha.
pub fn gp_predict_mean_cpu(
    k_star: &[f32],
    alpha: &[f32],
    n_star: usize,
    n_train: usize,
) -> Vec<f32> {
    let mut mean = vec![0.0f32; n_star];
    for i in 0..n_star {
        let mut sum = 0.0f32;
        for j in 0..n_train {
            sum += k_star[i * n_train + j] * alpha[j];
        }
        mean[i] = sum;
    }
    mean
}
