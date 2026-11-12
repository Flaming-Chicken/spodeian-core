//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 12:
//! 12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.

use crate::context::KernelDevice;

/// WGSL Compute Shader for parallel SVGD velocity field evaluation across particles:
/// phi(x_i) = (1/M) sum_j [k(x_j, x_i) grad log p(x_j) + (x_i - x_j)/h^2 * k(x_j, x_i)]
pub const SVGD_VELOCITY_SHADER: &str = r#"
struct SvgdUniforms {
    m_particles: u32,
    d_dim: u32,
    h_sq: f32,
    padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: SvgdUniforms;
@group(0) @binding(1) var<storage, read> particles: array<f32>;
@group(0) @binding(2) var<storage, read> grad_logp: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(4) var<storage, read_write> kernel_gram: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let m = uniforms.m_particles;
    let d = uniforms.d_dim;
    let h_sq = uniforms.h_sq;

    if (i >= m) {
        return;
    }

    let m_f32 = f32(m);

    for (var c: u32 = 0u; c < d; c = c + 1u) {
        var drive: f32 = 0.0;
        var repulsive: f32 = 0.0;

        for (var j: u32 = 0u; j < m; j = j + 1u) {
            // Compute squared distance on first dimension pass
            var dist_sq: f32 = 0.0;
            for (var k_dim: u32 = 0u; k_dim < d; k_dim = k_dim + 1u) {
                let diff_k = particles[i * d + k_dim] - particles[j * d + k_dim];
                dist_sq = dist_sq + diff_k * diff_k;
            }
            let k_val = exp(-0.5 * dist_sq / h_sq);
            if (c == 0u) {
                kernel_gram[i * m + j] = k_val;
            }

            drive = drive + k_val * grad_logp[j * d + c];
            let diff_c = particles[i * d + c] - particles[j * d + c];
            repulsive = repulsive + (diff_c / h_sq) * k_val;
        }

        velocity[i * d + c] = (drive + repulsive) / m_f32;
    }
}
"#;

/// Runs L symplectic leapfrog steps on quadratic potential U(theta) = 0.5 * theta^T Sigma_inv theta on CPU.
pub fn hmc_leapfrog_cpu(
    theta: &[f32],
    p: &[f32],
    sigma_inv: &[f32],
    eps: f32,
    l_steps: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, f32) {
    let mut cur_theta = theta.to_vec();
    let mut cur_p = p.to_vec();

    for _ in 0..l_steps {
        // Half step momentum: p = p - 0.5 * eps * (Sigma_inv @ cur_theta)
        for i in 0..d {
            let mut grad_u = 0.0f32;
            for j in 0..d {
                grad_u += sigma_inv[i * d + j] * cur_theta[j];
            }
            cur_p[i] -= 0.5f32 * eps * grad_u;
        }

        // Full step position: theta = theta + eps * cur_p
        for i in 0..d {
            cur_theta[i] += eps * cur_p[i];
        }

        // Half step momentum: p = p - 0.5 * eps * (Sigma_inv @ cur_theta)
        for i in 0..d {
            let mut grad_u = 0.0f32;
            for j in 0..d {
                grad_u += sigma_inv[i * d + j] * cur_theta[j];
            }
            cur_p[i] -= 0.5f32 * eps * grad_u;
        }
    }

    // Acceptance calculation
    let mut u_init = 0.0f32;
    let mut k_init = 0.0f32;
    for i in 0..d {
        let mut row_sum = 0.0f32;
        for j in 0..d {
            row_sum += sigma_inv[i * d + j] * theta[j];
        }
        u_init += theta[i] * row_sum;
        k_init += p[i] * p[i];
    }
    let h_init = 0.5f32 * (u_init + k_init);

    let mut u_prop = 0.0f32;
    let mut k_prop = 0.0f32;
    for i in 0..d {
        let mut row_sum = 0.0f32;
        for j in 0..d {
            row_sum += sigma_inv[i * d + j] * cur_theta[j];
        }
        u_prop += cur_theta[i] * row_sum;
        k_prop += cur_p[i] * cur_p[i];
    }
    let h_prop = 0.5f32 * (u_prop + k_prop);

    let accept = (h_init - h_prop).exp().min(1.0f32);
    (cur_theta, cur_p, accept)
}

/// Executes single SGLD step:
/// theta_{t+1} = theta_t + 0.5 * eps * grad_logp + sqrt(eps) * noise
pub fn sgld_step_cpu(theta: &[f32], grad_logp: &[f32], noise: &[f32], eps: f32) -> Vec<f32> {
    let d = theta.len();
    let sqrt_eps = eps.sqrt();
    let mut next_theta = vec![0.0f32; d];
    for i in 0..d {
        next_theta[i] = theta[i] + 0.5f32 * eps * grad_logp[i] + sqrt_eps * noise[i];
    }
    next_theta
}

/// Computes SVGD velocity field and RBF kernel Gram matrix on CPU.
pub fn svgd_velocity_field_cpu(
    particles: &[f32],
    grad_logp: &[f32],
    m: usize,
    d: usize,
    h: f32,
) -> (Vec<f32>, Vec<f32>) {
    let h_sq = h * h;
    let mut k_gram = vec![0.0f32; m * m];

    for i in 0..m {
        for j in 0..m {
            let mut dist_sq = 0.0f32;
            for c in 0..d {
                let diff = particles[i * d + c] - particles[j * d + c];
                dist_sq += diff * diff;
            }
            k_gram[i * m + j] = (-0.5f32 * dist_sq / h_sq).exp();
        }
    }

    let mut vel = vec![0.0f32; m * d];
    let m_f32 = m as f32;

    for i in 0..m {
        for c in 0..d {
            let mut drive = 0.0f32;
            let mut repulsive = 0.0f32;
            for j in 0..m {
                let k_ij = k_gram[i * m + j];
                drive += k_ij * grad_logp[j * d + c];
                let diff = particles[i * d + c] - particles[j * d + c];
                repulsive += (diff / h_sq) * k_ij;
            }
            vel[i * d + c] = (drive + repulsive) / m_f32;
        }
    }

    (vel, k_gram)
}

/// Dispatches SVGD velocity calculation to GPU if available; falls back to CPU.
pub fn svgd_velocity_field_auto(
    device: Option<&KernelDevice>,
    particles: &[f32],
    grad_logp: &[f32],
    m: usize,
    d: usize,
    h: f32,
) -> (Vec<f32>, Vec<f32>) {
    let dev = match device {
        Some(d_dev) => d_dev,
        None => return svgd_velocity_field_cpu(particles, grad_logp, m, d, h),
    };

    match dev {
        KernelDevice::Cpu => svgd_velocity_field_cpu(particles, grad_logp, m, d, h),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct SvgdUniforms {
                m_particles: u32,
                d_dim: u32,
                h_sq: f32,
                padding: f32,
            }

            let uniforms = SvgdUniforms {
                m_particles: m as u32,
                d_dim: d as u32,
                h_sq: h * h,
                padding: 0.0,
            };

            let uniform_buffer =
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SVGD Uniforms Buffer"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

            let particles_buffer =
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SVGD Particles Buffer"),
                    contents: bytemuck::cast_slice(particles),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

            let grad_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SVGD Grad Buffer"),
                contents: bytemuck::cast_slice(grad_logp),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let vel_size = (m * d * std::mem::size_of::<f32>()) as u64;
            let vel_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SVGD Velocity Buffer"),
                size: vel_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let gram_size = (m * m * std::mem::size_of::<f32>()) as u64;
            let gram_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SVGD Kernel Gram Buffer"),
                size: gram_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_vel = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Vel"),
                size: vel_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let staging_gram = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Gram"),
                size: gram_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("SVGD Velocity Shader"),
                source: wgpu::ShaderSource::Wgsl(SVGD_VELOCITY_SHADER.into()),
            });

            let bind_group_layout =
                wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SVGD Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    ],
                });

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SVGD Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particles_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grad_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: vel_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: gram_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout =
                wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SVGD Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SVGD Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVGD Encoder"),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SVGD Compute Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let workgroups = ((m as u32) + 63) / 64;
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&vel_buffer, 0, &staging_vel, 0, vel_size);
            encoder.copy_buffer_to_buffer(&gram_buffer, 0, &staging_gram, 0, gram_size);
            queue.submit(Some(encoder.finish()));

            let vel_slice = staging_vel.slice(..);
            let gram_slice = staging_gram.slice(..);

            let (tx_v, rx_v) = std::sync::mpsc::channel();
            vel_slice.map_async(wgpu::MapMode::Read, move |res| {
                tx_v.send(res).unwrap();
            });

            let (tx_g, rx_g) = std::sync::mpsc::channel();
            gram_slice.map_async(wgpu::MapMode::Read, move |res| {
                tx_g.send(res).unwrap();
            });

            wgpu_device.poll(wgpu::Maintain::Wait);
            if rx_v.recv().is_err() || rx_g.recv().is_err() {
                return svgd_velocity_field_cpu(particles, grad_logp, m, d, h);
            }

            let vel_data = vel_slice.get_mapped_range();
            let vel_vec: Vec<f32> = bytemuck::cast_slice(&vel_data).to_vec();
            drop(vel_data);
            staging_vel.unmap();

            let gram_data = gram_slice.get_mapped_range();
            let gram_vec: Vec<f32> = bytemuck::cast_slice(&gram_data).to_vec();
            drop(gram_data);
            staging_gram.unmap();

            (vel_vec, gram_vec)
        }
    }
}
