//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 10:
//! 10_survival_analysis_and_contextual_bandits.

use crate::context::KernelDevice;

/// WGSL Compute Shader for parallel LinUCB contextual arm scoring:
/// Computes score_a = x^T theta_a + alpha * sqrt(x^T A_inv_a x) for each arm in parallel.
pub const LINUCB_SCORING_SHADER: &str = r#"
struct LinucbUniforms {
    num_arms: u32,
    dim: u32,
    alpha: f32,
    padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: LinucbUniforms;
@group(0) @binding(1) var<storage, read> x_context: array<f32>;
@group(0) @binding(2) var<storage, read> theta_arms: array<f32>;
@group(0) @binding(3) var<storage, read> a_inv_arms: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let arm = global_id.x;
    if (arm >= uniforms.num_arms) {
        return;
    }

    let d = uniforms.dim;
    let theta_offset = arm * d;
    let a_inv_offset = arm * d * d;

    // mean_reward = x @ theta_arm
    var mean_reward: f32 = 0.0;
    for (var i: u32 = 0u; i < d; i = i + 1u) {
        mean_reward = mean_reward + x_context[i] * theta_arms[theta_offset + i];
    }

    // var = x @ A_inv @ x
    var quad_form: f32 = 0.0;
    for (var i: u32 = 0u; i < d; i = i + 1u) {
        var a_inv_x_i: f32 = 0.0;
        for (var j: u32 = 0u; j < d; j = j + 1u) {
            a_inv_x_i = a_inv_x_i + a_inv_arms[a_inv_offset + i * d + j] * x_context[j];
        }
        quad_form = quad_form + x_context[i] * a_inv_x_i;
    }

    let conf_width = uniforms.alpha * sqrt(max(quad_form, 0.0));
    scores[arm] = mean_reward + conf_width;
}
"#;

/// Inverts small NxN matrix using Gauss-Jordan elimination.
pub fn invert_matrix_cpu(matrix: &[f32], n: usize) -> Vec<f32> {
    let mut a = vec![vec![0.0f32; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = matrix[i * n + j];
        }
        a[i][n + i] = 1.0f32;
    }

    for i in 0..n {
        let mut pivot_row = i;
        let mut max_val = a[i][i].abs();
        for k in (i + 1)..n {
            if a[k][i].abs() > max_val {
                max_val = a[k][i].abs();
                pivot_row = k;
            }
        }
        assert!(max_val >= 1e-12, "Singular matrix");
        a.swap(i, pivot_row);

        let pivot = a[i][i];
        for j in 0..(2 * n) {
            a[i][j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = a[k][i];
                for j in 0..(2 * n) {
                    a[k][j] -= factor * a[i][j];
                }
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i][n + j];
        }
    }
    inv
}

/// Evaluates LinUCB exploration scores on CPU.
pub fn linucb_score_arms_cpu(
    x_context: &[f32],
    a_arms: &[f32],
    b_arms: &[f32],
    num_arms: usize,
    dim: usize,
    alpha: f32,
) -> (Vec<f32>, usize) {
    let mut scores = vec![0.0f32; num_arms];
    let mut best_arm = 0usize;
    let mut max_score = f32::NEG_INFINITY;

    for a in 0..num_arms {
        let a_slice = &a_arms[a * dim * dim..(a + 1) * dim * dim];
        let a_inv = invert_matrix_cpu(a_slice, dim);
        let b_slice = &b_arms[a * dim..(a + 1) * dim];

        let mut theta_hat = vec![0.0f32; dim];
        for i in 0..dim {
            let mut sum = 0.0f32;
            for j in 0..dim {
                sum += a_inv[i * dim + j] * b_slice[j];
            }
            theta_hat[i] = sum;
        }

        let mut mean_reward = 0.0f32;
        for i in 0..dim {
            mean_reward += x_context[i] * theta_hat[i];
        }

        let mut a_inv_x = vec![0.0f32; dim];
        for i in 0..dim {
            let mut sum = 0.0f32;
            for j in 0..dim {
                sum += a_inv[i * dim + j] * x_context[j];
            }
            a_inv_x[i] = sum;
        }

        let mut quad_form = 0.0f32;
        for i in 0..dim {
            quad_form += x_context[i] * a_inv_x[i];
        }

        let conf_width = alpha * quad_form.max(0.0).sqrt();
        let score = mean_reward + conf_width;
        scores[a] = score;
        if score > max_score {
            max_score = score;
            best_arm = a;
        }
    }

    (scores, best_arm)
}

/// Dispatches LinUCB scoring on GPU if available; falls back to CPU.
pub fn linucb_score_arms_auto(
    device: Option<&KernelDevice>,
    x_context: &[f32],
    a_arms: &[f32],
    b_arms: &[f32],
    num_arms: usize,
    dim: usize,
    alpha: f32,
) -> (Vec<f32>, usize) {
    let dev = match device {
        Some(d) => d,
        None => return linucb_score_arms_cpu(x_context, a_arms, b_arms, num_arms, dim, alpha),
    };

    match dev {
        KernelDevice::Cpu => linucb_score_arms_cpu(x_context, a_arms, b_arms, num_arms, dim, alpha),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            // Compute theta_arms and a_inv_arms on CPU for kernel dispatch
            let mut theta_arms = vec![0.0f32; num_arms * dim];
            let mut a_inv_arms = vec![0.0f32; num_arms * dim * dim];

            for a in 0..num_arms {
                let a_slice = &a_arms[a * dim * dim..(a + 1) * dim * dim];
                let a_inv = invert_matrix_cpu(a_slice, dim);
                let b_slice = &b_arms[a * dim..(a + 1) * dim];

                for i in 0..(dim * dim) {
                    a_inv_arms[a * dim * dim + i] = a_inv[i];
                }

                for i in 0..dim {
                    let mut sum = 0.0f32;
                    for j in 0..dim {
                        sum += a_inv[i * dim + j] * b_slice[j];
                    }
                    theta_arms[a * dim + i] = sum;
                }
            }

            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            let out_size = (num_arms * std::mem::size_of::<f32>()) as u64;

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct LinucbUniforms {
                num_arms: u32,
                dim: u32,
                alpha: f32,
                padding: f32,
            }

            let uniforms = LinucbUniforms {
                num_arms: num_arms as u32,
                dim: dim as u32,
                alpha,
                padding: 0.0,
            };

            let uniform_buffer =
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LinUCB Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let x_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LinUCB X Buffer"),
                contents: bytemuck::cast_slice(x_context),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let theta_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LinUCB Theta Buffer"),
                contents: bytemuck::cast_slice(&theta_arms),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let a_inv_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LinUCB A_inv Buffer"),
                contents: bytemuck::cast_slice(&a_inv_arms),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("LinUCB Out Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let readback_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("LinUCB Readback Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("LinUCB Shader"),
                source: wgpu::ShaderSource::Wgsl(LINUCB_SCORING_SHADER.into()),
            });

            let bind_group_layout =
                wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("LinUCB Bind Group Layout"),
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
                    ],
                });

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LinUCB Bind Group"),
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
                        resource: theta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: a_inv_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout =
                wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LinUCB Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LinUCB Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("LinUCB Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LinUCB Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let wg_x = (num_arms as u32).div_ceil(64);
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
                return linucb_score_arms_cpu(x_context, a_arms, b_arms, num_arms, dim, alpha);
            }

            let data = buffer_slice.get_mapped_range();
            let scores: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            readback_buffer.unmap();

            let mut best_arm = 0usize;
            let mut max_score = f32::NEG_INFINITY;
            for a in 0..num_arms {
                if scores[a] > max_score {
                    max_score = scores[a];
                    best_arm = a;
                }
            }

            (scores, best_arm)
        }
        #[cfg(not(feature = "wgpu"))]
        KernelDevice::Wgpu(_) => {
            linucb_score_arms_cpu(x_context, a_arms, b_arms, num_arms, dim, alpha)
        }
    }
}

/// Evaluates Kaplan-Meier survival estimator S(t) on CPU.
pub fn kaplan_meier_survival_cpu(times: &[f32], events: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut unique_times: Vec<f32> = times
        .iter()
        .zip(events.iter())
        .filter(|&(_, &e)| e == 1.0)
        .map(|(&t, _)| t)
        .collect();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_times.dedup();

    let mut surv_probs = Vec::with_capacity(unique_times.len());
    let mut current_s = 1.0f32;

    for &t_i in &unique_times {
        let mut n_i = 0.0f32;
        let mut d_i = 0.0f32;
        for i in 0..n {
            if times[i] >= t_i {
                n_i += 1.0;
            }
            if times[i] == t_i && events[i] == 1.0 {
                d_i += 1.0;
            }
        }
        current_s *= 1.0 - (d_i / n_i);
        surv_probs.push(current_s);
    }

    (unique_times, surv_probs)
}

/// Evaluates Cox Proportional Hazards negative partial log-likelihood on CPU.
pub fn cox_ph_loss_cpu(risk: &[f32], times: &[f32], events: &[f32], n: usize) -> f32 {
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| times[j].partial_cmp(&times[i]).unwrap());

    let mut log_risk_set = vec![0.0f32; n];
    let mut running_lse = f32::NEG_INFINITY;

    for i in 0..n {
        let r_val = risk[indices[i]];
        if running_lse == f32::NEG_INFINITY {
            running_lse = r_val;
        } else {
            let max_val = running_lse.max(r_val);
            let min_val = running_lse.min(r_val);
            running_lse = max_val + (1.0f32 + (min_val - max_val).exp()).ln();
        }
        log_risk_set[i] = running_lse;
    }

    let mut total_loss = 0.0f32;
    for i in 0..n {
        let idx = indices[i];
        if events[idx] == 1.0 {
            total_loss -= risk[idx] - log_risk_set[i];
        }
    }

    total_loss
}

/// Computes Sherman-Morrison rank-1 update for A_new^{-1} in O(d^2) time.
pub fn sherman_morrison_rank1_update_cpu(a_inv: &[f32], x: &[f32], dim: usize) -> Vec<f32> {
    let mut a_inv_x = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += a_inv[i * dim + j] * x[j];
        }
        a_inv_x[i] = sum;
    }

    let mut x_a_inv_x = 0.0f32;
    for i in 0..dim {
        x_a_inv_x += x[i] * a_inv_x[i];
    }
    let denom = 1.0f32 + x_a_inv_x;

    let mut out = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let outer = a_inv_x[i] * a_inv_x[j];
            out[i * dim + j] = a_inv[i * dim + j] - (outer / denom);
        }
    }
    out
}
