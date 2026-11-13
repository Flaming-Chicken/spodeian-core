//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 13:
//! 13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.

use crate::context::KernelDevice;

/// WGSL Compute Shader for parallel Clayton Copula and GPD Extreme Value evaluation across data points:
/// C(u, v) = max(u^(-theta) + v^(-theta) - 1, 0)^(-1/theta)
/// F(y) = 1 - (1 + xi * y / sigma)^(-1/xi)
pub const COPULA_GPD_SHADER: &str = r#"
struct CopulaUniforms {
    n_points: u32,
    theta_copula: f32,
    xi_gpd: f32,
    sigma_gpd: f32,
};

@group(0) @binding(0) var<uniform> uniforms: CopulaUniforms;
@group(0) @binding(1) var<storage, read> u_data: array<f32>;
@group(0) @binding(2) var<storage, read> v_data: array<f32>;
@group(0) @binding(3) var<storage, read> y_excesses: array<f32>;
@group(0) @binding(4) var<storage, read_write> copula_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> gpd_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.n_points) {
        return;
    }

    let theta = uniforms.theta_copula;
    let u = u_data[idx];
    let v = v_data[idx];

    // Clayton copula: (u^(-theta) + v^(-theta) - 1)^(-1/theta)
    let inner = max(pow(u, -theta) + pow(v, -theta) - 1.0, 1e-7);
    copula_out[idx] = pow(inner, -1.0 / theta);

    // GPD tail CDF: 1 - (1 + xi * y / sigma)^(-1/xi)
    let y = y_excesses[idx];
    let xi = uniforms.xi_gpd;
    let sigma = uniforms.sigma_gpd;
    let bracket = max(1.0 + (xi * y) / sigma, 1e-7);
    gpd_out[idx] = 1.0 - pow(bracket, -1.0 / xi);
}
"#;

/// Evaluates smooth and decomposable 4-variable SPN on CPU.
pub fn evaluate_spn_cpu(
    leaves_p: &[f32],      // [4 * 2]
    leaf_evidence: &[f32], // [4]
    prod_weights: &[f32],  // [2]
) -> f32 {
    let mut leaf_vals = [[1.0f32; 2]; 4];

    for i in 0..4 {
        let ev = leaf_evidence[i];
        if (ev - 1.0f32).abs() < 1e-5 {
            leaf_vals[i][0] = leaves_p[i * 2 + 0];
            leaf_vals[i][1] = leaves_p[i * 2 + 1];
        } else if ev.abs() < 1e-5 {
            leaf_vals[i][0] = 1.0f32 - leaves_p[i * 2 + 0];
            leaf_vals[i][1] = 1.0f32 - leaves_p[i * 2 + 1];
        } else {
            // Marginalized out
            leaf_vals[i][0] = 1.0f32;
            leaf_vals[i][1] = 1.0f32;
        }
    }

    let p0 = leaf_vals[0][0] * leaf_vals[1][0] * leaf_vals[2][0] * leaf_vals[3][0];
    let p1 = leaf_vals[0][1] * leaf_vals[1][1] * leaf_vals[2][1] * leaf_vals[3][1];
    prod_weights[0] * p0 + prod_weights[1] * p1
}

/// Evaluates Clayton copula CDF on CPU.
pub fn clayton_copula_cdf_cpu(u: &[f32], v: &[f32], theta: f32) -> Vec<f32> {
    let n = u.len();
    let mut out = vec![0.0f32; n];
    let neg_inv_theta = -1.0 / (theta as f64);

    for i in 0..n {
        let u_val = u[i] as f64;
        let v_val = v[i] as f64;
        let inner = (u_val.powf(-(theta as f64)) + v_val.powf(-(theta as f64)) - 1.0).max(1e-7);
        out[i] = inner.powf(neg_inv_theta) as f32;
    }
    out
}

/// Evaluates GPD tail CDF on CPU.
pub fn gpd_tail_cdf_cpu(y: &[f32], xi: f32, sigma: f32) -> Vec<f32> {
    let n = y.len();
    let mut out = vec![0.0f32; n];
    let neg_inv_xi = -1.0 / (xi as f64);

    for i in 0..n {
        let y_val = y[i] as f64;
        let bracket = (1.0 + ((xi as f64) * y_val) / (sigma as f64)).max(1e-7);
        out[i] = (1.0 - bracket.powf(neg_inv_xi)) as f32;
    }
    out
}

/// Evaluates SBI neural ratio and logit on CPU.
pub fn sbi_neural_ratio_cpu(
    x: &[f32],
    theta: &[f32],
    w: &[f32],
    b: &[f32],
    b_size: usize,
    d_x: usize,
    d_theta: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut ratio = vec![0.0f32; b_size];
    let mut logit = vec![0.0f32; b_size];

    for i in 0..b_size {
        let mut sum = b[0];
        for j in 0..d_x {
            sum += x[i * d_x + j] * w[j];
        }
        for j in 0..d_theta {
            sum += theta[i * d_theta + j] * w[d_x + j];
        }
        logit[i] = sum;
        ratio[i] = sum.exp();
    }

    (ratio, logit)
}

/// Dispatches Clayton Copula and GPD evaluation to GPU if available; falls back to CPU.
pub fn copula_gpd_auto(
    device: Option<&KernelDevice>,
    u: &[f32],
    v: &[f32],
    y: &[f32],
    theta: f32,
    xi: f32,
    sigma: f32,
) -> (Vec<f32>, Vec<f32>) {
    let dev = match device {
        Some(d) => d,
        None => {
            return (
                clayton_copula_cdf_cpu(u, v, theta),
                gpd_tail_cdf_cpu(y, xi, sigma),
            );
        }
    };

    match dev {
        KernelDevice::Cpu => (
            clayton_copula_cdf_cpu(u, v, theta),
            gpd_tail_cdf_cpu(y, xi, sigma),
        ),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let n = u.len();
            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct CopulaUniforms {
                n_points: u32,
                theta_copula: f32,
                xi_gpd: f32,
                sigma_gpd: f32,
            }

            let uniforms = CopulaUniforms {
                n_points: n as u32,
                theta_copula: theta,
                xi_gpd: xi,
                sigma_gpd: sigma,
            };

            let uniform_buffer =
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Copula Uniforms Buffer"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

            let u_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("U Buffer"),
                contents: bytemuck::cast_slice(u),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let v_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("V Buffer"),
                contents: bytemuck::cast_slice(v),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let y_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Y Buffer"),
                contents: bytemuck::cast_slice(y),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let out_size = (n * std::mem::size_of::<f32>()) as u64;

            let copula_out_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Copula Out Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let gpd_out_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPD Out Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_copula = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Copula"),
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let staging_gpd = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging GPD"),
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Copula GPD Shader"),
                source: wgpu::ShaderSource::Wgsl(COPULA_GPD_SHADER.into()),
            });

            let bind_group_layout =
                wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Copula Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Copula Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: y_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: copula_out_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: gpd_out_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout =
                wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Copula Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Copula Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copula Encoder"),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Copula Compute Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let workgroups = ((n as u32) + 63) / 64;
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&copula_out_buffer, 0, &staging_copula, 0, out_size);
            encoder.copy_buffer_to_buffer(&gpd_out_buffer, 0, &staging_gpd, 0, out_size);
            queue.submit(Some(encoder.finish()));

            let copula_slice = staging_copula.slice(..);
            let gpd_slice = staging_gpd.slice(..);

            let (tx_c, rx_c) = std::sync::mpsc::channel();
            copula_slice.map_async(wgpu::MapMode::Read, move |res| {
                tx_c.send(res).unwrap();
            });

            let (tx_g, rx_g) = std::sync::mpsc::channel();
            gpd_slice.map_async(wgpu::MapMode::Read, move |res| {
                tx_g.send(res).unwrap();
            });

            wgpu_device.poll(wgpu::Maintain::Wait);
            if rx_c.recv().is_err() || rx_g.recv().is_err() {
                return (
                    clayton_copula_cdf_cpu(u, v, theta),
                    gpd_tail_cdf_cpu(y, xi, sigma),
                );
            }

            let copula_data = copula_slice.get_mapped_range();
            let c_vec: Vec<f32> = bytemuck::cast_slice(&copula_data).to_vec();
            drop(copula_data);
            staging_copula.unmap();

            let gpd_data = gpd_slice.get_mapped_range();
            let g_vec: Vec<f32> = bytemuck::cast_slice(&gpd_data).to_vec();
            drop(gpd_data);
            staging_gpd.unmap();

            (c_vec, g_vec)
        }
    }
}
