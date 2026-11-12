//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 04:
//! 04_orthogonal_and_preconditioned_matrix_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Fused Muon Step:
/// param_out = param_in * (1 - lr * weight_decay) - lr * ortho_update
pub const MUON_UPDATE_SHADER: &str = r#"
struct MuonUniforms {
    len: u32,
    lr: f32,
    weight_decay: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: MuonUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> ortho_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> param_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }

    let p = param_in[idx];
    let o = ortho_in[idx];

    let shrunk = p * (1.0 - uniforms.lr * uniforms.weight_decay);
    param_out[idx] = shrunk - uniforms.lr * o;
}
"#;

/// CPU reference routine for Newton-Schulz quintic iteration.
pub fn newton_schulz_quintic_cpu(
    g: &[f32],
    m: usize,
    n: usize,
    steps: usize,
    eps: f32,
) -> Vec<f32> {
    let mut sum_sq = 0.0f32;
    for &val in g {
        sum_sq += val * val;
    }
    let norm_g = sum_sq.sqrt() + eps;

    let transpose = m >= n;
    let (rows, cols) = if transpose { (n, m) } else { (m, n) };

    let mut x = vec![0.0f32; rows * cols];
    if transpose {
        for r in 0..m {
            for c in 0..n {
                x[c * m + r] = g[r * n + c] / norm_g;
            }
        }
    } else {
        for i in 0..(m * n) {
            x[i] = g[i] / norm_g;
        }
    }

    let a_coeff = 3.4445f32;
    let b_coeff = -4.7750f32;
    let c_coeff = 2.0315f32;

    for _ in 0..steps {
        // A = X @ X^T: [rows, rows]
        let mut a = vec![0.0f32; rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = 0.0f32;
                for k in 0..cols {
                    sum += x[i * cols + k] * x[j * cols + k];
                }
                a[i * rows + j] = sum;
            }
        }

        // A^2 = A @ A: [rows, rows]
        let mut a_sq = vec![0.0f32; rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = 0.0f32;
                for k in 0..rows {
                    sum += a[i * rows + k] * a[k * rows + j];
                }
                a_sq[i * rows + j] = sum;
            }
        }

        // B_part = b * A + c * A^2: [rows, rows]
        let mut b_part = vec![0.0f32; rows * rows];
        for i in 0..(rows * rows) {
            b_part[i] = b_coeff * a[i] + c_coeff * a_sq[i];
        }

        // update = B_part @ X: [rows, cols]
        // new_X = a * X + update
        let mut new_x = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0f32;
                for k in 0..rows {
                    sum += b_part[i * rows + k] * x[k * cols + j];
                }
                new_x[i * cols + j] = a_coeff * x[i * cols + j] + sum;
            }
        }
        x = new_x;
    }

    if transpose {
        let mut res = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                res[r * n + c] = x[c * m + r];
            }
        }
        res
    } else {
        x
    }
}

/// CPU reference routine for Muon step.
pub fn muon_cpu(
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    out_param: &mut [f32],
    out_m: &mut [f32],
    m: usize,
    n: usize,
    lr: f32,
    momentum: f32,
    nesterov: bool,
    weight_decay: f32,
    ns_steps: usize,
) {
    let len = m * n;
    assert_eq!(param.len(), len);
    assert_eq!(grad.len(), len);
    assert_eq!(m_buf.len(), len);
    assert_eq!(out_param.len(), len);
    assert_eq!(out_m.len(), len);

    let mut update_cand = vec![0.0f32; len];
    for i in 0..len {
        let nm = momentum * m_buf[i] + grad[i];
        out_m[i] = nm;
        update_cand[i] = if nesterov {
            momentum * nm + grad[i]
        } else {
            nm
        };
    }

    let ortho = newton_schulz_quintic_cpu(&update_cand, m, n, ns_steps, 1e-7);

    for i in 0..len {
        let shrunk = param[i] * (1.0 - lr * weight_decay);
        out_param[i] = shrunk - lr * ortho[i];
    }
}

/// Auto-dispatcher executing Muon step with WGPU compute shader and CPU fallback.
pub fn muon_auto(
    device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    m: usize,
    n: usize,
    lr: f32,
    momentum: f32,
    nesterov: bool,
    weight_decay: f32,
    ns_steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let len = m * n;
    let mut out_param = vec![0.0f32; len];
    let mut out_m = vec![0.0f32; len];

    let mut update_cand = vec![0.0f32; len];
    for i in 0..len {
        let nm = momentum * m_buf[i] + grad[i];
        out_m[i] = nm;
        update_cand[i] = if nesterov {
            momentum * nm + grad[i]
        } else {
            nm
        };
    }
    let ortho = newton_schulz_quintic_cpu(&update_cand, m, n, ns_steps, 1e-7);

    #[cfg(feature = "wgpu")]
    if let KernelDevice::Wgpu(ctx) = device {
        if let Ok(res_p) = run_muon_wgpu(ctx, param, &ortho, lr, weight_decay) {
            return (res_p, out_m);
        }
    }

    for i in 0..len {
        let shrunk = param[i] * (1.0 - lr * weight_decay);
        out_param[i] = shrunk - lr * ortho[i];
    }

    (out_param, out_m)
}

#[cfg(feature = "wgpu")]
fn run_muon_wgpu(
    ctx: &crate::context::WgpuContext,
    param: &[f32],
    ortho: &[f32],
    lr: f32,
    weight_decay: f32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use wgpu::util::DeviceExt;

    let n = param.len();
    let size_bytes = (n * std::mem::size_of::<f32>()) as u64;

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Uniforms {
        len: u32,
        lr: f32,
        weight_decay: f32,
        _pad: f32,
    }

    let uniforms = Uniforms {
        len: n as u32,
        lr,
        weight_decay,
        _pad: 0.0,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("muon_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let p_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("muon_p_in"),
            contents: bytemuck::cast_slice(param),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let o_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("muon_o_in"),
            contents: bytemuck::cast_slice(ortho),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let p_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("muon_p_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cs_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("muon_shader"),
            source: wgpu::ShaderSource::Wgsl(MUON_UPDATE_SHADER.into()),
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("muon_layout"),
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

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("muon_pipe_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("muon_pipeline"),
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("muon_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: u_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: p_in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: o_in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: p_out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("muon_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("muon_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((n as u32) + 63) / 64;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    let p_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("muon_stage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&p_out_buf, 0, &p_stage, 0, size_bytes);
    ctx.queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    let p_slice = p_stage.slice(..);
    p_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()??;

    let res_p = bytemuck::cast_slice::<u8, f32>(&p_slice.get_mapped_range()).to_vec();
    p_stage.unmap();

    Ok(res_p)
}

/// Computes M^(-1/4) via Jacobi diagonalization for a symmetric matrix.
pub fn matrix_power_neg_fourth_cpu(m_mat: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let mut a = vec![vec![0.0f64; n]; n];
    let mut v = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        v[i][i] = 1.0;
        for j in 0..n {
            a[i][j] = m_mat[i * n + j] as f64;
        }
    }

    for _ in 0..60 {
        let mut p = 0;
        let mut q = 1;
        let mut max_val = 0.0f64;

        for i in 0..n {
            for j in (i + 1)..n {
                let abs_val = a[i][j].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i][p] = c * a[i][p] - s * a[i][q];
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = s * a[i][p] + c * a[i][q];
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = c * c * a[p][p] - 2.0 * s * c * a[p][q] + s * s * a[q][q];
        new_a[q][q] = s * s * a[p][p] + 2.0 * s * c * a[p][q] + c * c * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        for i in 0..n {
            let v_ip = v[i][p];
            let v_iq = v[i][q];
            v[i][p] = c * v_ip - s * v_iq;
            v[i][q] = s * v_ip + c * v_iq;
        }
    }

    let mut inv_d = vec![0.0f64; n];
    for i in 0..n {
        let val = a[i][i].max(eps as f64);
        inv_d[i] = val.powf(-0.25);
    }

    let mut res = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += v[i][k] * inv_d[k] * v[j][k];
            }
            res[i * n + j] = sum as f32;
        }
    }
    res
}

/// CPU reference routine for Shampoo step.
pub fn shampoo_cpu(
    param: &[f32],
    grad: &[f32],
    l_cov: &[f32],
    r_cov: &[f32],
    out_param: &mut [f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    m: usize,
    n: usize,
    lr: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) {
    // 1. Compute grad @ grad^T: [m, m]
    let mut g_gt = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += grad[i * n + k] * grad[j * n + k];
            }
            g_gt[i * m + j] = sum;
        }
    }

    // 2. Compute grad^T @ grad: [n, n]
    let mut gt_g = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..m {
                sum += grad[k * n + i] * grad[k * n + j];
            }
            gt_g[i * n + j] = sum;
        }
    }

    // 3. Update L and R
    for i in 0..(m * m) {
        out_l[i] = beta2 * l_cov[i] + (1.0 - beta2) * g_gt[i];
    }
    for i in 0..(n * n) {
        out_r[i] = beta2 * r_cov[i] + (1.0 - beta2) * gt_g[i];
    }

    // 4. Invert fourth root
    let l_inv = matrix_power_neg_fourth_cpu(out_l, m, eps);
    let r_inv = matrix_power_neg_fourth_cpu(out_r, n, eps);

    // 5. Preconditioned gradient: L_inv @ grad @ R_inv: [m, n]
    let mut tmp = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..m {
                sum += l_inv[i * m + k] * grad[k * n + j];
            }
            tmp[i * n + j] = sum;
        }
    }

    let mut precond = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += tmp[i * n + k] * r_inv[k * n + j];
            }
            precond[i * n + j] = sum;
        }
    }

    for i in 0..(m * n) {
        let shrunk = param[i] * (1.0 - lr * weight_decay);
        out_param[i] = shrunk - lr * precond[i];
    }
}

/// CPU reference routine for GaLore step.
pub fn galore_cpu(
    param: &[f32],
    grad: &[f32],
    m_low: &[f32],
    v_low: &[f32],
    proj: &[f32], // [m, r]
    out_param: &mut [f32],
    out_m: &mut [f32],
    out_v: &mut [f32],
    m: usize,
    n: usize,
    r: usize,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) {
    // grad_low = P^T @ grad: [r, n]
    let mut grad_low = vec![0.0f32; r * n];
    for i in 0..r {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..m {
                sum += proj[k * r + i] * grad[k * n + j];
            }
            grad_low[i * n + j] = sum;
        }
    }

    let bc1 = (1.0 - (beta1 as f64).powi(step as i32)) as f32;
    let bc2 = (1.0 - (beta2 as f64).powi(step as i32)) as f32;

    let mut adam_step = vec![0.0f32; r * n];
    for i in 0..(r * n) {
        let g = grad_low[i];
        let nm = beta1 * m_low[i] + (1.0 - beta1) * g;
        let nv = beta2 * v_low[i] + (1.0 - beta2) * (g * g);
        out_m[i] = nm;
        out_v[i] = nv;

        let m_hat = nm / bc1;
        let v_hat = nv / bc2;
        adam_step[i] = m_hat / (v_hat.sqrt() + eps);
    }

    // full_update = P @ adam_step: [m, n]
    let mut full_update = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..r {
                sum += proj[i * r + k] * adam_step[k * n + j];
            }
            full_update[i * n + j] = sum;
        }
    }

    for i in 0..(m * n) {
        let shrunk = param[i] * (1.0 - lr * weight_decay);
        out_param[i] = shrunk - lr * full_update[i];
    }
}
