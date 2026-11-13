//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 09:
//! 09_hidden_markov_models_kalman_ukf_and_particle_filters.

use crate::context::KernelDevice;

/// WGSL Compute Shader for parallel Unscented Sigma Points displacement:
/// Generates 2n+1 sigma points given mean mu and Cholesky matrix L.
pub const UKF_SIGMA_SHADER: &str = r#"
struct UkfUniforms {
    n_dim: u32,
    gamma: f32,
};

@group(0) @binding(0) var<uniform> uniforms: UkfUniforms;
@group(0) @binding(1) var<storage, read> mu: array<f32>;
@group(0) @binding(2) var<storage, read> l_mat: array<f32>;
@group(0) @binding(3) var<storage, read_write> sigma_pts: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pt_idx = global_id.x;
    let total_pts = 2u * uniforms.n_dim + 1u;
    if (pt_idx >= total_pts) {
        return;
    }

    let n = uniforms.n_dim;
    let offset = pt_idx * n;

    if (pt_idx == 0u) {
        // Point 0 is mu
        for (var d: u32 = 0u; d < n; d = d + 1u) {
            sigma_pts[offset + d] = mu[d];
        }
    } else if (pt_idx <= n) {
        // Points 1..n: mu + gamma * L[:, i - 1]
        let col = pt_idx - 1u;
        for (var d: u32 = 0u; d < n; d = d + 1u) {
            sigma_pts[offset + d] = mu[d] + uniforms.gamma * l_mat[d * n + col];
        }
    } else {
        // Points n+1..2n: mu - gamma * L[:, i - 1 - n]
        let col = pt_idx - 1u - n;
        for (var d: u32 = 0u; d < n; d = d + 1u) {
            sigma_pts[offset + d] = mu[d] - uniforms.gamma * l_mat[d * n + col];
        }
    }
}
"#;

/// Computes lower-triangular Cholesky factor L for SPD matrix.
pub fn cholesky_cpu(sigma: &[f32], n: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = sigma[i * n + i] - sum;
                assert!(
                    diag > 0.0,
                    "Covariance is not positive definite: diag = {diag}"
                );
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (sigma[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    l
}

/// Computes 2n+1 Merwe scaled unscented sigma points on CPU.
pub fn generate_ukf_sigma_points_cpu(
    mu: &[f32],
    sigma: &[f32],
    n_dim: usize,
    alpha: f32,
    _beta: f32,
    kappa: f32,
) -> Vec<f32> {
    let lambda_val = (alpha * alpha) * ((n_dim as f32) + kappa) - (n_dim as f32);
    let gamma = ((n_dim as f32) + lambda_val).sqrt();
    let l_mat = cholesky_cpu(sigma, n_dim);

    let num_pts = 2 * n_dim + 1;
    let mut out = vec![0.0f32; num_pts * n_dim];

    // Point 0: mu
    for d in 0..n_dim {
        out[d] = mu[d];
    }

    // Points 1..n: mu + gamma * L[:, i]
    for i in 0..n_dim {
        let offset = (i + 1) * n_dim;
        for d in 0..n_dim {
            out[offset + d] = mu[d] + gamma * l_mat[d * n_dim + i];
        }
    }

    // Points n+1..2n: mu - gamma * L[:, i]
    for i in 0..n_dim {
        let offset = (i + 1 + n_dim) * n_dim;
        for d in 0..n_dim {
            out[offset + d] = mu[d] - gamma * l_mat[d * n_dim + i];
        }
    }

    out
}

/// Dispatches UKF sigma point generation on GPU if available; falls back to CPU.
pub fn generate_ukf_sigma_points_auto(
    device: Option<&KernelDevice>,
    mu: &[f32],
    sigma: &[f32],
    n_dim: usize,
    alpha: f32,
    beta: f32,
    kappa: f32,
) -> Vec<f32> {
    let dev = match device {
        Some(d) => d,
        None => return generate_ukf_sigma_points_cpu(mu, sigma, n_dim, alpha, beta, kappa),
    };

    match dev {
        KernelDevice::Cpu => generate_ukf_sigma_points_cpu(mu, sigma, n_dim, alpha, beta, kappa),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let lambda_val = (alpha * alpha) * ((n_dim as f32) + kappa) - (n_dim as f32);
            let gamma = ((n_dim as f32) + lambda_val).sqrt();
            let l_mat = cholesky_cpu(sigma, n_dim);

            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            let total_pts = 2 * n_dim + 1;
            let out_size = (total_pts * n_dim * std::mem::size_of::<f32>()) as u64;

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct UkfUniforms {
                n_dim: u32,
                gamma: f32,
            }

            let uniforms = UkfUniforms {
                n_dim: n_dim as u32,
                gamma,
            };

            let uniform_buffer =
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("UKF Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let mu_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("UKF Mu Buffer"),
                contents: bytemuck::cast_slice(mu),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let l_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("UKF L Buffer"),
                contents: bytemuck::cast_slice(&l_mat),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("UKF Out Storage Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let readback_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("UKF Readback Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("UKF Shader"),
                source: wgpu::ShaderSource::Wgsl(UKF_SIGMA_SHADER.into()),
            });

            let bind_group_layout =
                wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("UKF Bind Group Layout"),
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
                label: Some("UKF Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: mu_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: l_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout =
                wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("UKF Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("UKF Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("UKF Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("UKF Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let wg_x = (total_pts as u32).div_ceil(64);
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
                return generate_ukf_sigma_points_cpu(mu, sigma, n_dim, alpha, beta, kappa);
            }

            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            readback_buffer.unmap();

            result
        }
        #[cfg(not(feature = "wgpu"))]
        KernelDevice::Wgpu(_) => {
            generate_ukf_sigma_points_cpu(mu, sigma, n_dim, alpha, beta, kappa)
        }
    }
}

/// Forward Linear Kalman Filter on CPU with Joseph-stabilized covariance updates.
pub fn kalman_filter_forward_cpu(
    y: &[f32],
    t_steps: usize,
    f: &[f32],
    h: &[f32],
    q: &[f32],
    r: &[f32],
    x0: &[f32],
    p0: &[f32],
    d_state: usize,
    d_obs: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut x_hat = x0.to_vec();
    let mut p_hat = p0.to_vec();

    let mut x_filt = vec![0.0f32; t_steps * d_state];
    let mut p_diag = vec![0.0f32; t_steps * d_state];

    // Helper: matrix multiply A (n x k) @ B (k x m) -> C (n x m)
    let matmul = |a: &[f32], b: &[f32], n: usize, k: usize, m: usize| -> Vec<f32> {
        let mut c = vec![0.0f32; n * m];
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * m + j];
                }
                c[i * m + j] = sum;
            }
        }
        c
    };

    // Helper: transpose
    let transpose = |a: &[f32], n: usize, m: usize| -> Vec<f32> {
        let mut at = vec![0.0f32; m * n];
        for i in 0..n {
            for j in 0..m {
                at[j * n + i] = a[i * m + j];
            }
        }
        at
    };

    for t in 0..t_steps {
        let y_t = &y[t * d_obs..(t + 1) * d_obs];

        // 1. Predict state: x_pred = F @ x_hat
        let mut x_pred = vec![0.0f32; d_state];
        for i in 0..d_state {
            for j in 0..d_state {
                x_pred[i] += f[i * d_state + j] * x_hat[j];
            }
        }

        // 2. Predict covariance: P_pred = F @ P_hat @ F^T + Q
        let f_t = transpose(f, d_state, d_state);
        let fp = matmul(f, &p_hat, d_state, d_state, d_state);
        let mut p_pred = matmul(&fp, &f_t, d_state, d_state, d_state);
        for i in 0..(d_state * d_state) {
            p_pred[i] += q[i];
        }

        // 3. Innovation: y_t - H @ x_pred
        let mut innov = vec![0.0f32; d_obs];
        for i in 0..d_obs {
            let mut hx = 0.0f32;
            for j in 0..d_state {
                hx += h[i * d_state + j] * x_pred[j];
            }
            innov[i] = y_t[i] - hx;
        }

        // 4. S = H @ P_pred @ H^T + R
        let h_t = transpose(h, d_obs, d_state);
        let hp = matmul(h, &p_pred, d_obs, d_state, d_state);
        let mut s = matmul(&hp, &h_t, d_obs, d_state, d_obs);
        for i in 0..(d_obs * d_obs) {
            s[i] += r[i];
        }

        // 5. Invert S (d_obs x d_obs)
        let s_inv = if d_obs == 1 {
            vec![1.0f32 / s[0]]
        } else {
            let det = s[0] * s[3] - s[1] * s[2];
            let inv_det = 1.0f32 / det;
            vec![
                s[3] * inv_det,
                -s[1] * inv_det,
                -s[2] * inv_det,
                s[0] * inv_det,
            ]
        };

        // 6. Kalman gain: K = P_pred @ H^T @ S^{-1}
        let ph_t = matmul(&p_pred, &h_t, d_state, d_state, d_obs);
        let k_mat = matmul(&ph_t, &s_inv, d_state, d_obs, d_obs);

        // 7. Update state: x_hat = x_pred + K @ innov
        for i in 0..d_state {
            let mut k_innov = 0.0f32;
            for j in 0..d_obs {
                k_innov += k_mat[i * d_obs + j] * innov[j];
            }
            x_hat[i] = x_pred[i] + k_innov;
        }

        // 8. Joseph stabilized covariance update: P = (I - KH) P_pred (I - KH)^T + K R K^T
        let kh = matmul(&k_mat, h, d_state, d_obs, d_state);
        let mut i_kh = vec![0.0f32; d_state * d_state];
        for i in 0..d_state {
            for j in 0..d_state {
                let eye = if i == j { 1.0f32 } else { 0.0f32 };
                i_kh[i * d_state + j] = eye - kh[i * d_state + j];
            }
        }
        let i_kh_t = transpose(&i_kh, d_state, d_state);
        let ikh_p = matmul(&i_kh, &p_pred, d_state, d_state, d_state);
        let p_part1 = matmul(&ikh_p, &i_kh_t, d_state, d_state, d_state);

        let k_t = transpose(&k_mat, d_state, d_obs);
        let kr = matmul(&k_mat, r, d_state, d_obs, d_obs);
        let p_part2 = matmul(&kr, &k_t, d_state, d_obs, d_state);

        for i in 0..(d_state * d_state) {
            p_hat[i] = p_part1[i] + p_part2[i];
        }

        // Save
        for d in 0..d_state {
            x_filt[t * d_state + d] = x_hat[d];
            p_diag[t * d_state + d] = p_hat[d * d_state + d];
        }
    }

    (x_filt, p_diag)
}

/// Log-domain LogSumExp helper.
fn logsumexp(vals: &[f32]) -> f32 {
    let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_val.is_infinite() {
        return f32::NEG_INFINITY;
    }
    let sum_exp: f32 = vals.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Evaluates discrete HMM total log likelihood using log-domain Forward algorithm.
pub fn hmm_forward_log_prob_cpu(
    obs: &[i64],
    t_steps: usize,
    log_pi: &[f32],
    log_a: &[f32],
    log_b: &[f32],
    s_states: usize,
    v_obs: usize,
) -> f32 {
    let mut alpha = vec![0.0f32; s_states];
    let o0 = obs[0] as usize;
    for s in 0..s_states {
        alpha[s] = log_pi[s] + log_b[s * v_obs + o0];
    }

    for t in 1..t_steps {
        let ot = obs[t] as usize;
        let mut next_alpha = vec![0.0f32; s_states];
        for s in 0..s_states {
            let mut trans_scores = vec![0.0f32; s_states];
            for prev in 0..s_states {
                trans_scores[prev] = alpha[prev] + log_a[prev * s_states + s];
            }
            next_alpha[s] = logsumexp(&trans_scores) + log_b[s * v_obs + ot];
        }
        alpha = next_alpha;
    }

    logsumexp(&alpha)
}

/// Computes most likely state sequence via Viterbi (max-product) algorithm on CPU.
pub fn hmm_viterbi_decode_cpu(
    obs: &[i64],
    t_steps: usize,
    log_pi: &[f32],
    log_a: &[f32],
    log_b: &[f32],
    s_states: usize,
    v_obs: usize,
) -> Vec<i64> {
    let mut viterbi_probs = vec![vec![0.0f32; s_states]; t_steps];
    let mut backpointers = vec![vec![0usize; s_states]; t_steps];

    let o0 = obs[0] as usize;
    for s in 0..s_states {
        viterbi_probs[0][s] = log_pi[s] + log_b[s * v_obs + o0];
    }

    for t in 1..t_steps {
        let ot = obs[t] as usize;
        for s in 0..s_states {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = 0usize;
            for prev in 0..s_states {
                let score = viterbi_probs[t - 1][prev] + log_a[prev * s_states + s];
                if score > best_score {
                    best_score = score;
                    best_prev = prev;
                }
            }
            backpointers[t][s] = best_prev;
            viterbi_probs[t][s] = best_score + log_b[s * v_obs + ot];
        }
    }

    let mut best_last = 0usize;
    let mut max_last_prob = f32::NEG_INFINITY;
    for s in 0..s_states {
        if viterbi_probs[t_steps - 1][s] > max_last_prob {
            max_last_prob = viterbi_probs[t_steps - 1][s];
            best_last = s;
        }
    }

    let mut path = vec![0i64; t_steps];
    path[t_steps - 1] = best_last as i64;
    for t in (0..t_steps - 1).rev() {
        path[t] = backpointers[t + 1][path[t + 1] as usize] as i64;
    }

    path
}
