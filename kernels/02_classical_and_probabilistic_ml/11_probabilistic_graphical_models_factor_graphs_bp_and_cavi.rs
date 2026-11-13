//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 11:
//! 11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.

use crate::context::KernelDevice;

/// WGSL Compute Shader for parallel CAVI responsibility estimation across data points:
/// r_ik = softmax_k (-0.5 * (x_i - mu_k)^2 / sigma_sq)
pub const CAVI_RESPONSIBILITIES_SHADER: &str = r#"
struct CaviUniforms {
    n_points: u32,
    k_components: u32,
    sigma_sq: f32,
    padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: CaviUniforms;
@group(0) @binding(1) var<storage, read> x_data: array<f32>;
@group(0) @binding(2) var<storage, read> mu_components: array<f32>;
@group(0) @binding(3) var<storage, read_write> responsibilities: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.n_points) {
        return;
    }

    let x = x_data[idx];
    let k = uniforms.k_components;
    let s2 = uniforms.sigma_sq;

    // Find max log_p for numerical stability
    var max_lp: f32 = -1e30;
    for (var c: u32 = 0u; c < k; c = c + 1u) {
        let diff = x - mu_components[c];
        let lp = -0.5 * (diff * diff) / s2;
        if (lp > max_lp) {
            max_lp = lp;
        }
    }

    // Compute sum of exponentials
    var sum_exp: f32 = 0.0;
    for (var c: u32 = 0u; c < k; c = c + 1u) {
        let diff = x - mu_components[c];
        let lp = -0.5 * (diff * diff) / s2;
        sum_exp = sum_exp + exp(lp - max_lp);
    }

    // Write normalized responsibilities
    let offset = idx * k;
    for (var c: u32 = 0u; c < k; c = c + 1u) {
        let diff = x - mu_components[c];
        let lp = -0.5 * (diff * diff) / s2;
        responsibilities[offset + c] = exp(lp - max_lp) / sum_exp;
    }
}
"#;

/// Evaluates CAVI responsibilities and ELBO on CPU.
pub fn cavi_gmm_step_cpu(x: &[f32], mu: &[f32], sigma_sq: f32) -> (Vec<f32>, f32) {
    let n = x.len();
    let k = mu.len();
    let mut resp = vec![0.0f32; n * k];
    let mut elbo = 0.0f32;

    for i in 0..n {
        let mut log_p = vec![0.0f32; k];
        let mut max_lp = f32::NEG_INFINITY;
        for c in 0..k {
            let diff = x[i] - mu[c];
            log_p[c] = -0.5f32 * (diff * diff) / sigma_sq;
            if log_p[c] > max_lp {
                max_lp = log_p[c];
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..k {
            sum_exp += (log_p[c] - max_lp).exp();
        }

        for c in 0..k {
            let r_ic = (log_p[c] - max_lp).exp() / sum_exp;
            resp[i * k + c] = r_ic;
            let log_r_ic = r_ic.max(1e-8).ln();
            elbo += r_ic * (log_p[c] - log_r_ic);
        }
    }

    (resp, elbo)
}

/// Dispatches CAVI on GPU if available; falls back to CPU.
pub fn cavi_gmm_step_auto(
    device: Option<&KernelDevice>,
    x: &[f32],
    mu: &[f32],
    sigma_sq: f32,
) -> (Vec<f32>, f32) {
    let dev = match device {
        Some(d) => d,
        None => return cavi_gmm_step_cpu(x, mu, sigma_sq),
    };

    match dev {
        KernelDevice::Cpu => cavi_gmm_step_cpu(x, mu, sigma_sq),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let n = x.len();
            let k = mu.len();

            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            let out_size = (n * k * std::mem::size_of::<f32>()) as u64;

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct CaviUniforms {
                n_points: u32,
                k_components: u32,
                sigma_sq: f32,
                padding: f32,
            }

            let uniforms = CaviUniforms {
                n_points: n as u32,
                k_components: k as u32,
                sigma_sq,
                padding: 0.0,
            };

            let uniform_buffer =
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CAVI Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let x_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CAVI X Buffer"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let mu_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CAVI Mu Buffer"),
                contents: bytemuck::cast_slice(mu),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CAVI Out Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let readback_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CAVI Readback Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("CAVI Shader"),
                source: wgpu::ShaderSource::Wgsl(CAVI_RESPONSIBILITIES_SHADER.into()),
            });

            let bind_group_layout =
                wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("CAVI Bind Group Layout"),
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
                    ],
                });

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("CAVI Bind Group"),
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
                        resource: mu_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout =
                wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("CAVI Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CAVI Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CAVI Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CAVI Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let wg_x = (n as u32).div_ceil(64);
                compute_pass.dispatch_workgroups(wg_x, 1, 1);
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
                return cavi_gmm_step_cpu(x, mu, sigma_sq);
            }

            let data = buffer_slice.get_mapped_range();
            let resp: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            readback_buffer.unmap();

            // Compute ELBO from responsibilities
            let mut elbo = 0.0f32;
            for i in 0..n {
                for c in 0..k {
                    let diff = x[i] - mu[c];
                    let log_p = -0.5f32 * (diff * diff) / sigma_sq;
                    let r_ic = resp[i * k + c];
                    let log_r_ic = r_ic.max(1e-8).ln();
                    elbo += r_ic * (log_p - log_r_ic);
                }
            }

            (resp, elbo)
        }
        #[cfg(not(feature = "wgpu"))]
        KernelDevice::Wgpu(_) => cavi_gmm_step_cpu(x, mu, sigma_sq),
    }
}

/// Computes exact marginals for a 4-node tree via Sum-Product Belief Propagation on CPU.
pub fn sum_product_tree_bp_cpu(
    unary: &[f32], // 4 * 2 = 8
    edges: &[f32], // 3 * 2 * 2 = 12
) -> Vec<f32> {
    let u = |node: usize, val: usize| unary[node * 2 + val];
    let e = |edge_idx: usize, r: usize, c: usize| edges[edge_idx * 4 + r * 2 + c];

    // Messages leaves -> root (1):
    let m_0_1 = [
        u(0, 0) * e(0, 0, 0) + u(0, 1) * e(0, 1, 0),
        u(0, 0) * e(0, 0, 1) + u(0, 1) * e(0, 1, 1),
    ];

    let m_2_1 = [
        e(1, 0, 0) * u(2, 0) + e(1, 0, 1) * u(2, 1),
        e(1, 1, 0) * u(2, 0) + e(1, 1, 1) * u(2, 1),
    ];

    let m_3_1 = [
        e(2, 0, 0) * u(3, 0) + e(2, 0, 1) * u(3, 1),
        e(2, 1, 0) * u(3, 0) + e(2, 1, 1) * u(3, 1),
    ];

    // Belief at root 1:
    let b1_raw = [
        u(1, 0) * m_0_1[0] * m_2_1[0] * m_3_1[0],
        u(1, 1) * m_0_1[1] * m_2_1[1] * m_3_1[1],
    ];
    let z1 = b1_raw[0] + b1_raw[1];
    let b1 = [b1_raw[0] / z1, b1_raw[1] / z1];

    // Messages root 1 -> leaves:
    let prod_1_to_0 = [u(1, 0) * m_2_1[0] * m_3_1[0], u(1, 1) * m_2_1[1] * m_3_1[1]];
    let m_1_0 = [
        e(0, 0, 0) * prod_1_to_0[0] + e(0, 0, 1) * prod_1_to_0[1],
        e(0, 1, 0) * prod_1_to_0[0] + e(0, 1, 1) * prod_1_to_0[1],
    ];
    let b0_raw = [u(0, 0) * m_1_0[0], u(0, 1) * m_1_0[1]];
    let z0 = b0_raw[0] + b0_raw[1];
    let b0 = [b0_raw[0] / z0, b0_raw[1] / z0];

    let prod_1_to_2 = [u(1, 0) * m_0_1[0] * m_3_1[0], u(1, 1) * m_0_1[1] * m_3_1[1]];
    let m_1_2 = [
        prod_1_to_2[0] * e(1, 0, 0) + prod_1_to_2[1] * e(1, 1, 0),
        prod_1_to_2[0] * e(1, 0, 1) + prod_1_to_2[1] * e(1, 1, 1),
    ];
    let b2_raw = [u(2, 0) * m_1_2[0], u(2, 1) * m_1_2[1]];
    let z2 = b2_raw[0] + b2_raw[1];
    let b2 = [b2_raw[0] / z2, b2_raw[1] / z2];

    let prod_1_to_3 = [u(1, 0) * m_0_1[0] * m_2_1[0], u(1, 1) * m_0_1[1] * m_2_1[1]];
    let m_1_3 = [
        prod_1_to_3[0] * e(2, 0, 0) + prod_1_to_3[1] * e(2, 1, 0),
        prod_1_to_3[0] * e(2, 0, 1) + prod_1_to_3[1] * e(2, 1, 1),
    ];
    let b3_raw = [u(3, 0) * m_1_3[0], u(3, 1) * m_1_3[1]];
    let z3 = b3_raw[0] + b3_raw[1];
    let b3 = [b3_raw[0] / z3, b3_raw[1] / z3];

    vec![b0[0], b0[1], b1[0], b1[1], b2[0], b2[1], b3[0], b3[1]]
}

/// Computes MAP state configuration on tree using exact maximization on CPU.
pub fn max_product_tree_map_cpu(unary: &[f32], edges: &[f32]) -> Vec<i64> {
    let u = |node: usize, val: usize| unary[node * 2 + val];
    let e = |edge_idx: usize, r: usize, c: usize| edges[edge_idx * 4 + r * 2 + c];

    let mut best_config = vec![0i64; 4];
    let mut max_p = -1.0f32;

    for x0 in 0..2 {
        for x1 in 0..2 {
            for x2 in 0..2 {
                for x3 in 0..2 {
                    let p = u(0, x0)
                        * u(1, x1)
                        * u(2, x2)
                        * u(3, x3)
                        * e(0, x0, x1)
                        * e(1, x1, x2)
                        * e(2, x1, x3);
                    if p > max_p {
                        max_p = p;
                        best_config = vec![x0 as i64, x1 as i64, x2 as i64, x3 as i64];
                    }
                }
            }
        }
    }

    best_config
}

/// Evaluates Damped Loopy Belief Propagation on a 3-node cyclic graph on CPU.
pub fn damped_loopy_bp_triangle_cpu(
    unary: &[f32],
    edges: &[f32],
    alpha: f32,
    iters: usize,
) -> Vec<f32> {
    let u = |node: usize, val: usize| unary[node * 2 + val];
    let e = |edge_idx: usize, r: usize, c: usize| edges[edge_idx * 4 + r * 2 + c];

    let mut m = vec![vec![0.5f32; 2]; 6];

    for _ in 0..iters {
        // 0 -> 1: (u[0] * m[4]) @ e[0]
        let p0 = [u(0, 0) * m[4][0], u(0, 1) * m[4][1]];
        let raw_0_1 = [
            p0[0] * e(0, 0, 0) + p0[1] * e(0, 1, 0),
            p0[0] * e(0, 0, 1) + p0[1] * e(0, 1, 1),
        ];
        let z_0_1 = raw_0_1[0] + raw_0_1[1];
        m[0][0] = (1.0 - alpha) * m[0][0] + alpha * (raw_0_1[0] / z_0_1);
        m[0][1] = (1.0 - alpha) * m[0][1] + alpha * (raw_0_1[1] / z_0_1);

        // 1 -> 0: e[0] @ (u[1] * m[3])
        let p1 = [u(1, 0) * m[3][0], u(1, 1) * m[3][1]];
        let raw_1_0 = [
            e(0, 0, 0) * p1[0] + e(0, 0, 1) * p1[1],
            e(0, 1, 0) * p1[0] + e(0, 1, 1) * p1[1],
        ];
        let z_1_0 = raw_1_0[0] + raw_1_0[1];
        m[1][0] = (1.0 - alpha) * m[1][0] + alpha * (raw_1_0[0] / z_1_0);
        m[1][1] = (1.0 - alpha) * m[1][1] + alpha * (raw_1_0[1] / z_1_0);

        // 1 -> 2: (u[1] * m[0]) @ e[1]
        let p1_2 = [u(1, 0) * m[0][0], u(1, 1) * m[0][1]];
        let raw_1_2 = [
            p1_2[0] * e(1, 0, 0) + p1_2[1] * e(1, 1, 0),
            p1_2[0] * e(1, 0, 1) + p1_2[1] * e(1, 1, 1),
        ];
        let z_1_2 = raw_1_2[0] + raw_1_2[1];
        m[2][0] = (1.0 - alpha) * m[2][0] + alpha * (raw_1_2[0] / z_1_2);
        m[2][1] = (1.0 - alpha) * m[2][1] + alpha * (raw_1_2[1] / z_1_2);

        // 2 -> 1: e[1] @ (u[2] * m[5])
        let p2 = [u(2, 0) * m[5][0], u(2, 1) * m[5][1]];
        let raw_2_1 = [
            e(1, 0, 0) * p2[0] + e(1, 0, 1) * p2[1],
            e(1, 1, 0) * p2[0] + e(1, 1, 1) * p2[1],
        ];
        let z_2_1 = raw_2_1[0] + raw_2_1[1];
        m[3][0] = (1.0 - alpha) * m[3][0] + alpha * (raw_2_1[0] / z_2_1);
        m[3][1] = (1.0 - alpha) * m[3][1] + alpha * (raw_2_1[1] / z_2_1);

        // 2 -> 0: (u[2] * m[2]) @ e[2]
        let p2_0 = [u(2, 0) * m[2][0], u(2, 1) * m[2][1]];
        let raw_2_0 = [
            p2_0[0] * e(2, 0, 0) + p2_0[1] * e(2, 1, 0),
            p2_0[0] * e(2, 0, 1) + p2_0[1] * e(2, 1, 1),
        ];
        let z_2_0 = raw_2_0[0] + raw_2_0[1];
        m[4][0] = (1.0 - alpha) * m[4][0] + alpha * (raw_2_0[0] / z_2_0);
        m[4][1] = (1.0 - alpha) * m[4][1] + alpha * (raw_2_0[1] / z_2_0);

        // 0 -> 2: e[2] @ (u[0] * m[1])
        let p0_2 = [u(0, 0) * m[1][0], u(0, 1) * m[1][1]];
        let raw_0_2 = [
            e(2, 0, 0) * p0_2[0] + e(2, 0, 1) * p0_2[1],
            e(2, 1, 0) * p0_2[0] + e(2, 1, 1) * p0_2[1],
        ];
        let z_0_2 = raw_0_2[0] + raw_0_2[1];
        m[5][0] = (1.0 - alpha) * m[5][0] + alpha * (raw_0_2[0] / z_0_2);
        m[5][1] = (1.0 - alpha) * m[5][1] + alpha * (raw_0_2[1] / z_0_2);
    }

    let b0_raw = [u(0, 0) * m[1][0] * m[4][0], u(0, 1) * m[1][1] * m[4][1]];
    let z0 = b0_raw[0] + b0_raw[1];
    let b0 = [b0_raw[0] / z0, b0_raw[1] / z0];

    let b1_raw = [u(1, 0) * m[0][0] * m[3][0], u(1, 1) * m[0][1] * m[3][1]];
    let z1 = b1_raw[0] + b1_raw[1];
    let b1 = [b1_raw[0] / z1, b1_raw[1] / z1];

    let b2_raw = [u(2, 0) * m[2][0] * m[5][0], u(2, 1) * m[2][1] * m[5][1]];
    let z2 = b2_raw[0] + b2_raw[1];
    let b2 = [b2_raw[0] / z2, b2_raw[1] / z2];

    vec![b0[0], b0[1], b1[0], b1[1], b2[0], b2[1]]
}
