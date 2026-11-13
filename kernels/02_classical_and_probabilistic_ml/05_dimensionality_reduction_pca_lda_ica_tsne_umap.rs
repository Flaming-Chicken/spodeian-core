//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 05:
//! 05_dimensionality_reduction_pca_lda_ica_tsne_umap.

use crate::context::KernelDevice;

/// WGSL Compute Shader for column-wise mean centering: X_centered[i, d] = X[i, d] - mean[d].
pub const MEAN_CENTER_SHADER: &str = r#"
struct CenterUniforms {
    n: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> uniforms: CenterUniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read_write> centered_out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= uniforms.n || col >= uniforms.dim) {
        return;
    }

    let idx = row * uniforms.dim + col;
    centered_out[idx] = x[idx] - mean[col];
}
"#;

/// CPU reference calculation of column means: mean[d] = 1/N sum_i X[i, d].
pub fn compute_column_means_cpu(x: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; dim];
    for i in 0..n {
        for d in 0..dim {
            mean[d] += x[i * dim + d];
        }
    }
    for d in 0..dim {
        mean[d] /= n as f32;
    }
    mean
}

/// CPU reference calculation of mean-centered data: X_centered[i, d] = X[i, d] - mean[d].
pub fn center_matrix_cpu(x: &[f32], n: usize, dim: usize, mean: &[f32]) -> Vec<f32> {
    let mut centered = vec![0.0f32; n * dim];
    for i in 0..n {
        for d in 0..dim {
            centered[i * dim + d] = x[i * dim + d] - mean[d];
        }
    }
    centered
}

/// CPU reference calculation of PCA projection: proj = X_centered @ V_top.
pub fn pca_project_cpu(
    x_centered: &[f32],
    n: usize,
    dim: usize,
    v_top: &[f32],
    k_components: usize,
) -> Vec<f32> {
    let mut proj = vec![0.0f32; n * k_components];
    for i in 0..n {
        for c in 0..k_components {
            let mut sum = 0.0f32;
            for d in 0..dim {
                sum += x_centered[i * dim + d] * v_top[d * k_components + c];
            }
            proj[i * k_components + c] = sum;
        }
    }
    proj
}

/// Solves symmetric positive-definite linear system A x = b via Cholesky decomposition L L^T = A.
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

/// Solves Fisher's LDA direction w_lda = Sw^{-1} (mu1 - mu0).
pub fn solve_lda_direction_cpu(
    x_centered: &[f32],
    y: &[f32],
    n: usize,
    dim: usize,
    reg: f32,
) -> Vec<f32> {
    let mut n0 = 0.0f32;
    let mut n1 = 0.0f32;
    let mut mu0 = vec![0.0f32; dim];
    let mut mu1 = vec![0.0f32; dim];

    for i in 0..n {
        if y[i] < 0.5f32 {
            n0 += 1.0f32;
            for d in 0..dim {
                mu0[d] += x_centered[i * dim + d];
            }
        } else {
            n1 += 1.0f32;
            for d in 0..dim {
                mu1[d] += x_centered[i * dim + d];
            }
        }
    }

    for d in 0..dim {
        mu0[d] /= n0.max(1.0f32);
        mu1[d] /= n1.max(1.0f32);
    }

    let mut sw = vec![0.0f32; dim * dim];
    for i in 0..n {
        let m = if y[i] < 0.5f32 { &mu0 } else { &mu1 };
        for r in 0..dim {
            let diff_r = x_centered[i * dim + r] - m[r];
            for c in 0..dim {
                let diff_c = x_centered[i * dim + c] - m[c];
                sw[r * dim + c] += diff_r * diff_c;
            }
        }
    }

    for d in 0..dim {
        sw[d * dim + d] += reg;
    }

    let mut diff_mu = vec![0.0f32; dim];
    for d in 0..dim {
        diff_mu[d] = mu1[d] - mu0[d];
    }

    let w = solve_cholesky_spd(&sw, &diff_mu, dim);
    let mut norm_sq = 0.0f32;
    for d in 0..dim {
        norm_sq += w[d] * w[d];
    }
    let norm = norm_sq.sqrt().max(1e-8f32);

    let mut w_norm = vec![0.0f32; dim];
    for d in 0..dim {
        w_norm[d] = w[d] / norm;
    }
    w_norm
}

/// Executes mean-centering using either WGPU hardware or CPU fallback.
pub fn compute_center_matrix_auto(
    device: &KernelDevice,
    x: &[f32],
    n: usize,
    dim: usize,
    mean: &[f32],
) -> Vec<f32> {
    match device {
        KernelDevice::Cpu => center_matrix_cpu(x, n, dim, mean),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct Uniforms {
                n: u32,
                dim: u32,
                _pad0: u32,
                _pad1: u32,
            }

            let uniforms = Uniforms {
                n: n as u32,
                dim: dim as u32,
                _pad0: 0,
                _pad1: 0,
            };

            let u_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Center Uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let x_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Center X"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let m_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Center Mean"),
                contents: bytemuck::cast_slice(mean),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_len = n * dim;
            let out_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Center Output"),
                size: (out_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let read_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Center Staging Read"),
                size: (out_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let sm = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Center WGSL"),
                source: wgpu::ShaderSource::Wgsl(MEAN_CENTER_SHADER.into()),
            });

            let bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Center BGL"),
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
                label: Some("Center BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: m_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });

            let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Center PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let cp = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Center CP"),
                layout: Some(&pl),
                module: &sm,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut enc = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Center Encoder"),
            });

            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Center Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cp);
                pass.set_bind_group(0, &bg, &[]);
                let wg_x = (n as u32).div_ceil(16);
                let wg_y = (dim as u32).div_ceil(16);
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
        KernelDevice::Wgpu(_) => center_matrix_cpu(x, n, dim, mean),
    }
}
