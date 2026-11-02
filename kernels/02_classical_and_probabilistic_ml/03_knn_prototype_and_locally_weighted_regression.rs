//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 03:
//! 03_knn_prototype_and_locally_weighted_regression.

use crate::context::KernelDevice;

/// WGSL Compute Shader for pairwise squared Euclidean distance evaluation.
/// Computes D_ij = ||x1_i - x2_j||^2 with clamp(dist_sq, 0.0).
pub const PAIRWISE_DIST_SHADER: &str = r#"
struct DistUniforms {
    n1: u32,
    n2: u32,
    dim: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> uniforms: DistUniforms;
@group(0) @binding(1) var<storage, read> x1: array<f32>;
@group(0) @binding(2) var<storage, read> x2: array<f32>;
@group(0) @binding(3) var<storage, read_write> dist_out: array<f32>;

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
    dist_out[out_idx] = max(dist_sq, 0.0f);
}
"#;

/// CPU reference calculation of pairwise squared Euclidean distances: ||x1_i - x2_j||^2.
pub fn pairwise_sq_dist_cpu(x1: &[f32], n1: usize, x2: &[f32], n2: usize, dim: usize) -> Vec<f32> {
    let mut dist = vec![0.0f32; n1 * n2];
    for i in 0..n1 {
        for j in 0..n2 {
            let mut sum_sq = 0.0f32;
            for d in 0..dim {
                let diff = x1[i * dim + d] - x2[j * dim + d];
                sum_sq += diff * diff;
            }
            dist[i * n2 + j] = sum_sq.max(0.0f32);
        }
    }
    dist
}

/// Distance-weighted k-NN continuous regression: pred_j = sum_{k-nearest i} w_i y_i / sum w_i where w_i = 1 / (d_i + eps).
pub fn knn_regress_cpu(
    x_train: &[f32],
    y_train: &[f32],
    n_train: usize,
    x_test: &[f32],
    n_test: usize,
    dim: usize,
    k: usize,
    eps: f32,
) -> Vec<f32> {
    let dist_sq = pairwise_sq_dist_cpu(x_train, n_train, x_test, n_test, dim);
    let mut preds = Vec::with_capacity(n_test);

    for j in 0..n_test {
        let mut d_pairs: Vec<(f32, usize)> = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let d = dist_sq[i * n_test + j].sqrt();
            d_pairs.push((d, i));
        }

        d_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut sum_w = 0.0f32;
        let mut sum_wy = 0.0f32;
        for &(d, idx) in d_pairs.iter().take(k) {
            let w = 1.0f32 / (d + eps);
            sum_w += w;
            sum_wy += w * y_train[idx];
        }

        preds.push(if sum_w > 1e-8 { sum_wy / sum_w } else { 0.0f32 });
    }

    preds
}

/// Nadaraya-Watson kernel regression: y(x) = sum_i K_h(x, x_i) y_i / sum_i K_h(x, x_i).
pub fn nadaraya_watson_regress_cpu(
    x_train: &[f32],
    y_train: &[f32],
    n_train: usize,
    x_test: &[f32],
    n_test: usize,
    dim: usize,
    bandwidth: f32,
) -> Vec<f32> {
    let dist_sq = pairwise_sq_dist_cpu(x_train, n_train, x_test, n_test, dim);
    let factor = -0.5f32 / (bandwidth * bandwidth);

    let mut preds = Vec::with_capacity(n_test);

    for j in 0..n_test {
        let mut sum_w = 0.0f32;
        let mut sum_wy = 0.0f32;
        for i in 0..n_train {
            let d_sq = dist_sq[i * n_test + j];
            let w = (d_sq * factor).exp();
            sum_w += w;
            sum_wy += w * y_train[i];
        }
        preds.push(if sum_w > 1e-8 { sum_wy / sum_w } else { 0.0f32 });
    }

    preds
}

/// Executes pairwise squared distance calculation using either WGPU hardware or CPU fallback.
pub fn compute_pairwise_sq_dist_auto(
    device: &KernelDevice,
    x1: &[f32],
    n1: usize,
    x2: &[f32],
    n2: usize,
    dim: usize,
) -> Vec<f32> {
    match device {
        KernelDevice::Cpu => pairwise_sq_dist_cpu(x1, n1, x2, n2, dim),
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
            }

            let uniforms = Uniforms {
                n1: n1 as u32,
                n2: n2 as u32,
                dim: dim as u32,
                _pad0: 0,
            };

            let u_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dist Uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let x1_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dist x1"),
                contents: bytemuck::cast_slice(x1),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let x2_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dist x2"),
                contents: bytemuck::cast_slice(x2),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_len = n1 * n2;
            let out_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dist Output"),
                size: (out_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let read_buf = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dist Staging Read"),
                size: (out_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let sm = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dist WGSL"),
                source: wgpu::ShaderSource::Wgsl(PAIRWISE_DIST_SHADER.into()),
            });

            let bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Dist BGL"),
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
                label: Some("Dist BG"),
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
                label: Some("Dist PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let cp = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Dist CP"),
                layout: Some(&pl),
                module: &sm,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut enc = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dist Encoder"),
            });

            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dist Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cp);
                pass.set_bind_group(0, &bg, &[]);
                let wg_x = ((n1 as u32) + 15) / 16;
                let wg_y = ((n2 as u32) + 15) / 16;
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
        KernelDevice::Wgpu(_) => pairwise_sq_dist_cpu(x1, n1, x2, n2, dim),
    }
}
