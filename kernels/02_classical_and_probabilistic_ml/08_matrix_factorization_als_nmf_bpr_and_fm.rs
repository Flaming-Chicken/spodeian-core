//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 08:
//! 08_matrix_factorization_als_nmf_bpr_and_fm.

use crate::context::KernelDevice;

/// WGSL Compute Shader for 2-way Factorization Machine evaluation.
pub const FM_SHADER: &str = r#"
struct FmUniforms {
    n: u32,
    dim: u32,
    k_factors: u32,
    w0: f32,
};

@group(0) @binding(0) var<uniform> uniforms: FmUniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> fm_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.n) {
        return;
    }

    var linear_term = uniforms.w0;
    for (var d: u32 = 0u; d < uniforms.dim; d = d + 1u) {
        linear_term = linear_term + x[idx * uniforms.dim + d] * w[d];
    }

    var interaction_term: f32 = 0.0;
    for (var f: u32 = 0u; f < uniforms.k_factors; f = f + 1u) {
        var sum_xv: f32 = 0.0;
        var sum_x2v2: f32 = 0.0;
        for (var d: u32 = 0u; d < uniforms.dim; d = d + 1u) {
            let x_val = x[idx * uniforms.dim + d];
            let v_val = v[d * uniforms.k_factors + f];
            sum_xv = sum_xv + x_val * v_val;
            sum_x2v2 = sum_x2v2 + (x_val * x_val) * (v_val * v_val);
        }
        interaction_term = interaction_term + 0.5 * (sum_xv * sum_xv - sum_x2v2);
    }

    fm_out[idx] = linear_term + interaction_term;
}
"#;

/// CPU reference calculation for 2-way Factorization Machine.
pub fn factorization_machine_cpu(
    x: &[f32],
    n: usize,
    dim: usize,
    k_factors: usize,
    w0: f32,
    w: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let mut linear = w0;
        for d in 0..dim {
            linear += x[i * dim + d] * w[d];
        }

        let mut interaction = 0.0f32;
        for f in 0..k_factors {
            let mut sum_xv = 0.0f32;
            let mut sum_x2v2 = 0.0f32;
            for d in 0..dim {
                let x_val = x[i * dim + d];
                let v_val = v[d * k_factors + f];
                sum_xv += x_val * v_val;
                sum_x2v2 += (x_val * x_val) * (v_val * v_val);
            }
            interaction += 0.5f32 * (sum_xv * sum_xv - sum_x2v2);
        }

        out[i] = linear + interaction;
    }

    out
}

/// Dispatches FM computation on GPU if available; otherwise falls back to CPU.
pub fn factorization_machine_auto(
    device: Option<&KernelDevice>,
    x: &[f32],
    n: usize,
    dim: usize,
    k_factors: usize,
    w0: f32,
    w: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let dev = match device {
        Some(d) => d,
        None => return factorization_machine_cpu(x, n, dim, k_factors, w0, w, v),
    };

    match dev {
        KernelDevice::Cpu => factorization_machine_cpu(x, n, dim, k_factors, w0, w, v),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            let out_size = (n * std::mem::size_of::<f32>()) as u64;

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct FmUniforms {
                n: u32,
                dim: u32,
                k_factors: u32,
                w0: f32,
            }

            let uniforms = FmUniforms {
                n: n as u32,
                dim: dim as u32,
                k_factors: k_factors as u32,
                w0,
            };

            let uniform_buffer =
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FM Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let x_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FM X Buffer"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let w_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FM W Buffer"),
                contents: bytemuck::cast_slice(w),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let v_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FM V Buffer"),
                contents: bytemuck::cast_slice(v),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FM Out Storage Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let readback_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FM Readback Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("FM Shader"),
                source: wgpu::ShaderSource::Wgsl(FM_SHADER.into()),
            });

            let bind_group_layout =
                wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("FM Bind Group Layout"),
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
                label: Some("FM Bind Group"),
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
                        resource: w_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout =
                wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("FM Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("FM Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FM Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FM Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let wg_x = ((n + 63) / 64) as u32;
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
                return factorization_machine_cpu(x, n, dim, k_factors, w0, w, v);
            }

            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            readback_buffer.unmap();

            result
        }
        #[cfg(not(feature = "wgpu"))]
        KernelDevice::Wgpu(_) => factorization_machine_cpu(x, n, dim, k_factors, w0, w, v),
    }
}

/// Low-rank matrix multiplication P @ Q^T on CPU.
pub fn matrix_factorization_predict_cpu(
    p: &[f32],
    q: &[f32],
    u: usize,
    i_dim: usize,
    k_factors: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; u * i_dim];
    for row in 0..u {
        for col in 0..i_dim {
            let mut sum = 0.0f32;
            for f in 0..k_factors {
                sum += p[row * k_factors + f] * q[col * k_factors + f];
            }
            out[row * i_dim + col] = sum;
        }
    }
    out
}
