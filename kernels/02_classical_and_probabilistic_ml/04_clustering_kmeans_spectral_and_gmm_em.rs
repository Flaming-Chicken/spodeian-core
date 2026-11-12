//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 04:
//! 04_clustering_kmeans_spectral_and_gmm_em.

use crate::context::KernelDevice;

/// WGSL Compute Shader for k-Means distance and cluster assignment.
/// Each thread processes one sample x_i and finds nearest centroid.
pub const KMEANS_ASSIGN_SHADER: &str = r#"
struct KMeansUniforms {
    n: u32,
    k: u32,
    dim: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> uniforms: KMeansUniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> centers: array<f32>;
@group(0) @binding(3) var<storage, read_write> assignments: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.n) {
        return;
    }

    var min_dist: f32 = 1e30f;
    var best_k: f32 = 0.0f;

    for (var c: u32 = 0u; c < uniforms.k; c = c + 1u) {
        var dist_sq: f32 = 0.0f;
        for (var d: u32 = 0u; d < uniforms.dim; d = d + 1u) {
            let diff = x[idx * uniforms.dim + d] - centers[c * uniforms.dim + d];
            dist_sq = dist_sq + diff * diff;
        }
        if (dist_sq < min_dist) {
            min_dist = dist_sq;
            best_k = f32(c);
        }
    }

    assignments[idx] = best_k;
}
"#;

/// CPU reference calculation of k-Means cluster assignments: argmin_k ||x_i - c_k||^2.
pub fn kmeans_assign_cpu(x: &[f32], n: usize, centers: &[f32], k: usize, dim: usize) -> Vec<f32> {
    let mut assigns = Vec::with_capacity(n);
    for i in 0..n {
        let mut min_dist = f32::MAX;
        let mut best_k = 0f32;
        for c in 0..k {
            let mut dist_sq = 0.0f32;
            for d in 0..dim {
                let diff = x[i * dim + d] - centers[c * dim + d];
                dist_sq += diff * diff;
            }
            if dist_sq < min_dist {
                min_dist = dist_sq;
                best_k = c as f32;
            }
        }
        assigns.push(best_k);
    }
    assigns
}

/// GMM E-step: Computes log-domain posterior responsibilities.
pub fn gmm_e_step_cpu(
    x: &[f32],
    n: usize,
    mu: &[f32],
    var: &[f32],
    pi: &[f32],
    k: usize,
    dim: usize,
) -> Vec<f32> {
    let log_2pi = 1.837877f32;
    let mut resp = vec![0.0f32; n * k];

    for i in 0..n {
        let mut max_log = f32::NEG_INFINITY;
        for c in 0..k {
            let mut quad = 0.0f32;
            let mut log_det = 0.0f32;
            for d in 0..dim {
                let diff = x[i * dim + d] - mu[c * dim + d];
                let v = var[c * dim + d];
                quad += (diff * diff) / v;
                log_det += v.ln();
            }
            let log_p = -0.5f32 * ((dim as f32) * log_2pi + log_det + quad);
            let val = (pi[c] + 1e-8f32).ln() + log_p;
            resp[i * k + c] = val;
            if val > max_log {
                max_log = val;
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..k {
            sum_exp += (resp[i * k + c] - max_log).exp();
        }
        let log_sum = max_log + sum_exp.ln();

        for c in 0..k {
            resp[i * k + c] = (resp[i * k + c] - log_sum).exp();
        }
    }
    resp
}

/// Computes symmetric normalized graph Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}.
pub fn compute_symmetric_laplacian_cpu(x: &[f32], n: usize, dim: usize, gamma: f32) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    let mut degrees = vec![0.0f32; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let mut dist_sq = 0.0f32;
                for d in 0..dim {
                    let diff = x[i * dim + d] - x[j * dim + d];
                    dist_sq += diff * diff;
                }
                let val = (-gamma * dist_sq).exp();
                a[i * n + j] = val;
                degrees[i] += val;
            }
        }
    }

    let mut d_inv_sqrt = vec![0.0f32; n];
    for i in 0..n {
        let deg = degrees[i].max(1e-6f32);
        d_inv_sqrt[i] = 1.0f32 / deg.sqrt();
    }

    let mut l_sym = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let norm_a = d_inv_sqrt[i] * a[i * n + j] * d_inv_sqrt[j];
            let delta = if i == j { 1.0f32 } else { 0.0f32 };
            l_sym[i * n + j] = delta - norm_a;
        }
    }
    l_sym
}

/// Executes k-Means cluster assignment using either WGPU hardware or CPU fallback.
pub fn compute_kmeans_assign_auto(
    device: &KernelDevice,
    x: &[f32],
    n: usize,
    centers: &[f32],
    k: usize,
    dim: usize,
) -> Vec<f32> {
    match device {
        KernelDevice::Cpu => kmeans_assign_cpu(x, n, centers, k, dim),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct Uniforms {
                n: u32,
                k: u32,
                dim: u32,
                _pad0: u32,
            }

            let uniforms = Uniforms {
                n: n as u32,
                k: k as u32,
                dim: dim as u32,
                _pad0: 0,
            };

            let u_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("KMeans Uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let x_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("KMeans X"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let c_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("KMeans Centers"),
                contents: bytemuck::cast_slice(centers),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("KMeans Assign Output"),
                size: (n * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let read_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("KMeans Staging Read"),
                size: (n * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let sm = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("KMeans WGSL"),
                source: wgpu::ShaderSource::Wgsl(KMEANS_ASSIGN_SHADER.into()),
            });

            let bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("KMeans BGL"),
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
                label: Some("KMeans BG"),
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
                        resource: c_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });

            let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("KMeans PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let cp = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("KMeans CP"),
                layout: Some(&pl),
                module: &sm,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut enc = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("KMeans Encoder"),
            });

            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("KMeans Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cp);
                pass.set_bind_group(0, &bg, &[]);
                let wg_x = ((n as u32) + 63) / 64;
                pass.dispatch_workgroups(wg_x, 1, 1);
            }

            enc.copy_buffer_to_buffer(
                &out_buf,
                0,
                &read_buf,
                0,
                (n * std::mem::size_of::<f32>()) as u64,
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
        KernelDevice::Wgpu(_) => kmeans_assign_cpu(x, n, centers, k, dim),
    }
}
