//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 07:
//! 07_differentiable_trees_node_soft_gbdt_and_gam_nam.

use crate::context::KernelDevice;

/// WGSL Compute Shader for single depth=2 oblivious tree evaluation.
pub const NODE_TREE_SHADER: &str = r#"
struct NodeUniforms {
    n: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> uniforms: NodeUniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w_split: array<f32>;
@group(0) @binding(3) var<storage, read> b_split: array<f32>;
@group(0) @binding(4) var<storage, read> leaf_values: array<f32>;
@group(0) @binding(5) var<storage, read_write> tree_out: array<f32>;

fn sigmoid_fn(val: f32) -> f32 {
    return 1.0 / (1.0 + exp(-val));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.n) {
        return;
    }

    var z0: f32 = b_split[0];
    var z1: f32 = b_split[1];
    for (var d: u32 = 0u; d < uniforms.dim; d = d + 1u) {
        let x_val = x[idx * uniforms.dim + d];
        z0 = z0 + x_val * w_split[d];
        z1 = z1 + x_val * w_split[uniforms.dim + d];
    }

    let p0 = sigmoid_fn(z0);
    let p1 = sigmoid_fn(z1);

    let w0 = (1.0 - p0) * (1.0 - p1);
    let w1 = (1.0 - p0) * p1;
    let w2 = p0 * (1.0 - p1);
    let w3 = p0 * p1;

    tree_out[idx] = w0 * leaf_values[0] + w1 * leaf_values[1] + w2 * leaf_values[2] + w3 * leaf_values[3];
}
"#;

/// CPU reference calculation of depth=2 NODE ensemble output.
pub fn node_forward_cpu(
    x: &[f32],
    n: usize,
    dim: usize,
    w_split: &[f32],
    b_split: &[f32],
    leaf_values: &[f32],
    n_trees: usize,
) -> Vec<f32> {
    let mut ensemble_out = vec![0.0f32; n];
    let depth = 2;
    let n_leaves = 4;

    for t in 0..n_trees {
        let w_offset = t * depth * dim;
        let b_offset = t * depth;
        let leaf_offset = t * n_leaves;

        for i in 0..n {
            let mut z0 = b_split[b_offset];
            let mut z1 = b_split[b_offset + 1];
            for d in 0..dim {
                let x_val = x[i * dim + d];
                z0 += x_val * w_split[w_offset + d];
                z1 += x_val * w_split[w_offset + dim + d];
            }

            let p0 = 1.0f32 / (1.0f32 + (-z0).exp());
            let p1 = 1.0f32 / (1.0f32 + (-z1).exp());

            let w0 = (1.0f32 - p0) * (1.0f32 - p1);
            let w1 = (1.0f32 - p0) * p1;
            let w2 = p0 * (1.0f32 - p1);
            let w3 = p0 * p1;

            let val = w0 * leaf_values[leaf_offset]
                + w1 * leaf_values[leaf_offset + 1]
                + w2 * leaf_values[leaf_offset + 2]
                + w3 * leaf_values[leaf_offset + 3];

            ensemble_out[i] += val;
        }
    }

    ensemble_out
}

/// Executes single tree NODE evaluation on GPU if available; otherwise falls back to CPU.
pub fn node_single_tree_auto(
    device: Option<&KernelDevice>,
    x: &[f32],
    n: usize,
    dim: usize,
    w_split: &[f32],
    b_split: &[f32],
    leaf_values: &[f32],
) -> Vec<f32> {
    let dev = match device {
        Some(d) => d,
        None => return node_forward_cpu(x, n, dim, w_split, b_split, leaf_values, 1),
    };

    match dev {
        KernelDevice::Cpu => node_forward_cpu(x, n, dim, w_split, b_split, leaf_values, 1),
        #[cfg(feature = "wgpu")]
        KernelDevice::Wgpu(ctx) => {
            let wgpu_device = &ctx.device;
            let queue = &ctx.queue;
            use wgpu::util::DeviceExt;

            let out_size = (n * std::mem::size_of::<f32>()) as u64;

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct NodeUniforms {
                n: u32,
                dim: u32,
                _pad0: u32,
                _pad1: u32,
            }

            let uniforms = NodeUniforms {
                n: n as u32,
                dim: dim as u32,
                _pad0: 0,
                _pad1: 0,
            };

            let uniform_buffer =
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("NODE Uniform Buffer"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let x_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NODE X Buffer"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let w_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NODE W Buffer"),
                contents: bytemuck::cast_slice(w_split),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let b_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NODE B Buffer"),
                contents: bytemuck::cast_slice(b_split),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let leaves_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NODE Leaves Buffer"),
                contents: bytemuck::cast_slice(leaf_values),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let out_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NODE Out Storage Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let readback_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NODE Readback Buffer"),
                size: out_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("NODE Tree Shader"),
                source: wgpu::ShaderSource::Wgsl(NODE_TREE_SHADER.into()),
            });

            let bind_group_layout =
                wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("NODE Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                label: Some("NODE Bind Group"),
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
                        resource: b_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: leaves_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout =
                wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("NODE Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("NODE Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("NODE Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("NODE Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let wg_x = n.div_ceil(64) as u32;
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
                return node_forward_cpu(x, n, dim, w_split, b_split, leaf_values, 1);
            }

            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            readback_buffer.unmap();

            result
        }
        #[cfg(not(feature = "wgpu"))]
        KernelDevice::Wgpu(_) => node_forward_cpu(x, n, dim, w_split, b_split, leaf_values, 1),
    }
}

/// CPU reference calculation for Neural Additive Model (NAM).
pub fn nam_forward_cpu(
    x: &[f32],
    n: usize,
    dim: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    bias0: f32,
    hidden_dim: usize,
) -> Vec<f32> {
    let mut out = vec![bias0; n];

    for j in 0..dim {
        let w1_offset = j * hidden_dim;
        let b1_offset = j * hidden_dim;
        let w2_offset = j * hidden_dim;
        let b2_val = b2[j];

        for i in 0..n {
            let x_val = x[i * dim + j];
            let mut out_j = b2_val;

            for h in 0..hidden_dim {
                let lin = x_val * w1[w1_offset + h] + b1[b1_offset + h];
                let act = if lin > 0.0f32 { lin } else { 0.0f32 };
                out_j += act * w2[w2_offset + h];
            }

            out[i] += out_j;
        }
    }

    out
}

/// CPU reference implementation of Pool-Adjacent-Violators (PAV) monotonic regression.
pub fn pav_isotonic_regression_cpu(y: &[f32]) -> Vec<f32> {
    let mut blocks: Vec<(usize, usize, f32, f32)> =
        (0..y.len()).map(|i| (i, i, 1.0f32, y[i])).collect();

    let mut i = 0;
    while i + 1 < blocks.len() {
        let mean_curr = blocks[i].3 / blocks[i].2;
        let mean_next = blocks[i + 1].3 / blocks[i + 1].2;

        if mean_curr > mean_next {
            blocks[i].1 = blocks[i + 1].1;
            blocks[i].2 += blocks[i + 1].2;
            blocks[i].3 += blocks[i + 1].3;
            blocks.remove(i + 1);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }

    let mut out = vec![0.0f32; y.len()];
    for b in blocks {
        let val = b.3 / b.2;
        for idx in b.0..=b.1 {
            out[idx] = val;
        }
    }
    out
}
