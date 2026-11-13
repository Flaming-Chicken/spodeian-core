//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 02:
//! 02_support_vector_machines_and_kernel_methods.

use crate::context::KernelDevice;

/// WGSL Compute Shader for pairwise RBF Gaussian kernel evaluation.
/// Workgroup size (16, 16) computes tile of Gram matrix K_ij = exp(-gamma * ||x_i - x_j||^2).
pub const RBF_KERNEL_SHADER: &str = r#"
struct RbfUniforms {
    n1: u32,
    n2: u32,
    dim: u32,
    _pad0: u32,
    gamma: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> uniforms: RbfUniforms;
@group(0) @binding(1) var<storage, read> x1: array<f32>;
@group(0) @binding(2) var<storage, read> x2: array<f32>;
@group(0) @binding(3) var<storage, read_write> gram_out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= uniforms.n1 || col >= uniforms.n2) {
        return;
    }

    var dist_sq: f32 = 0.0f;
    for (var d: u32 = 0u; d < uniforms.dim; d = d + 1u) {
        let diff = x1[row * uniforms.dim + d] - x2[col * uniforms.dim + d];
        dist_sq = dist_sq + diff * diff;
    }

    let out_idx = row * uniforms.n2 + col;
    gram_out[out_idx] = exp(-uniforms.gamma * dist_sq);
}
"#;

/// CPU reference calculation of pairwise RBF kernel: exp(-gamma * ||x1_i - x2_j||^2).
pub fn rbf_kernel_cpu(
    x1: &[f32],
    n1: usize,
    x2: &[f32],
    n2: usize,
    dim: usize,
    gamma: f32,
) -> Vec<f32> {
    let mut gram = vec![0.0f32; n1 * n2];
    for i in 0..n1 {
        for j in 0..n2 {
            let mut dist_sq = 0.0f32;
            for d in 0..dim {
                let diff = x1[i * dim + d] - x2[j * dim + d];
                dist_sq += diff * diff;
            }
            gram[i * n2 + j] = (-gamma * dist_sq).exp();
        }
    }
    gram
}

/// CPU reference calculation of pairwise Polynomial kernel: (x1_i^T x2_j / dim + coef0)^degree.
pub fn poly_kernel_cpu(
    x1: &[f32],
    n1: usize,
    x2: &[f32],
    n2: usize,
    dim: usize,
    degree: f32,
    coef0: f32,
) -> Vec<f32> {
    let mut gram = vec![0.0f32; n1 * n2];
    for i in 0..n1 {
        for j in 0..n2 {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += x1[i * dim + d] * x2[j * dim + d];
            }
            let scaled = dot / (dim as f32) + coef0;
            gram[i * n2 + j] = scaled.powf(degree);
        }
    }
    gram
}

/// CPU Cholesky SPD solver A x = b.
pub fn solve_cholesky_spd(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let val = a[i * n + i] - sum;
                l[i * n + j] = if val > 1e-12f32 { val.sqrt() } else { 1e-6f32 };
            } else {
                let diag = l[j * n + j];
                l[i * n + j] = if diag.abs() > 1e-12f32 {
                    (a[i * n + j] - sum) / diag
                } else {
                    0.0f32
                };
            }
        }
    }

    // Forward solve: L y = b
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for k in 0..i {
            sum += l[i * n + k] * y[k];
        }
        let diag = l[i * n + i];
        y[i] = if diag.abs() > 1e-12f32 {
            (b[i] - sum) / diag
        } else {
            0.0f32
        };
    }

    // Back solve: L^T x = y
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = 0.0f32;
        for k in (i + 1)..n {
            sum += l[k * n + i] * x[k];
        }
        let diag = l[i * n + i];
        x[i] = if diag.abs() > 1e-12f32 {
            (y[i] - sum) / diag
        } else {
            0.0f32
        };
    }

    x
}

/// Solves Kernel Ridge Regression: (K + alpha * I) c = y.
pub fn solve_kernel_ridge_cpu(
    x: &[f32],
    y: &[f32],
    n: usize,
    dim: usize,
    gamma: f32,
    alpha: f32,
) -> Vec<f32> {
    let mut k = rbf_kernel_cpu(x, n, x, n, dim, gamma);
    for i in 0..n {
        k[i * n + i] += alpha;
    }
    solve_cholesky_spd(&k, y, n)
}

/// Computes Nyström low-rank Gram approximation: K_nm @ pinv(K_mm) @ K_nm.T.
pub fn nystrom_approximate_gram_cpu(
    x: &[f32],
    n: usize,
    landmarks: &[f32],
    m: usize,
    dim: usize,
    gamma: f32,
    eps: f32,
) -> Vec<f32> {
    let mut k_mm = rbf_kernel_cpu(landmarks, m, landmarks, m, dim, gamma);
    for i in 0..m {
        k_mm[i * m + i] += eps;
    }
    let k_nm = rbf_kernel_cpu(x, n, landmarks, m, dim, gamma);

    // Invert k_mm column by column
    let mut k_mm_inv = vec![0.0f32; m * m];
    for col in 0..m {
        let mut e = vec![0.0f32; m];
        e[col] = 1.0f32;
        let col_sol = solve_cholesky_spd(&k_mm, &e, m);
        for row in 0..m {
            k_mm_inv[row * m + col] = col_sol[row];
        }
    }

    // temp = K_nm @ K_mm_inv: [n, m]
    let mut temp = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0f32;
            for k in 0..m {
                sum += k_nm[i * m + k] * k_mm_inv[k * m + j];
            }
            temp[i * m + j] = sum;
        }
    }

    // result = temp @ K_nm.T: [n, n]
    let mut res = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..m {
                sum += temp[i * m + k] * k_nm[j * m + k];
            }
            res[i * n + j] = sum;
        }
    }
    res
}

/// Executes RBF kernel computation using either WGPU hardware or CPU fallback.
pub fn compute_rbf_kernel_auto(
    device: &KernelDevice,
    x1: &[f32],
    n1: usize,
    x2: &[f32],
    n2: usize,
    dim: usize,
    gamma: f32,
) -> Vec<f32> {
    match device {
        KernelDevice::Cpu => rbf_kernel_cpu(x1, n1, x2, n2, dim, gamma),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct Uniforms {
                n1: u32,
                n2: u32,
                dim: u32,
                _pad0: u32,
                gamma: f32,
                _pad1: f32,
                _pad2: f32,
                _pad3: f32,
            }

            let uniforms = Uniforms {
                n1: n1 as u32,
                n2: n2 as u32,
                dim: dim as u32,
                _pad0: 0,
                gamma,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            };

            let u_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RBF Uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let x1_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RBF x1"),
                contents: bytemuck::cast_slice(x1),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let x2_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RBF x2"),
                contents: bytemuck::cast_slice(x2),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_len = n1 * n2;
            let out_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RBF Gram Output"),
                size: (out_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let read_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RBF Staging Read"),
                size: (out_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let sm = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RBF Kernel WGSL"),
                source: wgpu::ShaderSource::Wgsl(RBF_KERNEL_SHADER.into()),
            });

            let bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RBF BGL"),
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

            let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RBF BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x1_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: x2_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });

            let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RBF PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let cp = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RBF CP"),
                layout: Some(&pl),
                module: &sm,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut enc = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RBF Encoder"),
            });

            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RBF Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cp);
                pass.set_bind_group(0, &bg, &[]);
                let wg_x = (n1 as u32).div_ceil(16);
                let wg_y = (n2 as u32).div_ceil(16);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            enc.copy_buffer_to_buffer(
                &out_buf,
                0,
                &read_buf,
                0,
                (out_len * std::mem::size_of::<f32>()) as u64,
            );
            queue.submit(Some(enc.finish()));

            let slice = read_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                tx.send(res).unwrap();
            });
            wgpu_device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            let data = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            read_buf.unmap();
            result
        }
        #[cfg(not(feature = "wgpu"))]
        KernelDevice::Wgpu(_) => rbf_kernel_cpu(x1, n1, x2, n2, dim, gamma),
    }
}
