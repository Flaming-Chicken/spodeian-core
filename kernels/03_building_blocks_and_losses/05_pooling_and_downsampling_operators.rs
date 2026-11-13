//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 05:
//! Pooling and Downsampling Operators (GeM, BlurPool, PixelUnshuffle, and PMA).

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for Generalized Mean Pooling (GeM).
pub const GEM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<f32>;

struct Uniforms {
    b: u32,
    c: u32,
    h: u32,
    w: u32,
    p: f32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_channels = params.b * params.c;
    if (idx >= total_channels) {
        return;
    }
    let bi = idx / params.c;
    let ci = idx % params.c;
    let spatial_count = f32(params.h * params.w);
    let channel_offset = (bi * params.c + ci) * (params.h * params.w);

    var sum_p: f32 = 0.0;
    for (var i: u32 = 0u; i < params.h * params.w; i = i + 1u) {
        let val = max(in_data[channel_offset + i], params.eps);
        sum_p = sum_p + pow(val, params.p);
    }
    let mean_p = sum_p / spatial_count;
    out_data[idx] = pow(mean_p, 1.0 / params.p);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GeMUniforms {
    b: u32,
    c: u32,
    h: u32,
    w: u32,
    p: f32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Linear anchor forward on CPU: Y = X @ W + b.
pub fn linear_anchor_forward_cpu(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch_size: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * out_dim];
    for b in 0..batch_size {
        for j in 0..out_dim {
            let mut sum = bias[j];
            for i in 0..in_dim {
                sum += x[b * in_dim + i] * weight[i * out_dim + j];
            }
            out[b * out_dim + j] = sum;
        }
    }
    out
}

/// Generalized Mean Pooling (GeM) forward on CPU: (1/|R| sum x^p)^(1/p) over spatial dimensions (H, W).
pub fn gem_pool_cpu(
    x: &[f32],
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    p: f32,
    eps: f32,
) -> Vec<f32> {
    let spatial_count = (h * w) as f32;
    let mut out = vec![0.0f32; b * c];
    for bi in 0..b {
        for ci in 0..c {
            let channel_offset = (bi * c + ci) * (h * w);
            let mut sum_p = 0.0f32;
            for i in 0..(h * w) {
                let val = x[channel_offset + i].max(eps);
                sum_p += val.powf(p);
            }
            let mean_p = sum_p / spatial_count;
            out[bi * c + ci] = mean_p.powf(1.0 / p);
        }
    }
    out
}

/// Anti-Aliased Downsampling (BlurPool) on CPU with reflect padding and stride 2.
pub fn blur_pool_cpu(
    x: &[f32],
    kernel: &[f32], // [c * 1 * 3 * 3]
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    stride: usize,
) -> Vec<f32> {
    let pad_h = h + 2;
    let pad_w = w + 2;
    let mut padded = vec![0.0f32; b * c * pad_h * pad_w];

    for bi in 0..b {
        for ci in 0..c {
            let in_base = (bi * c + ci) * (h * w);
            let pad_base = (bi * c + ci) * (pad_h * pad_w);

            // Copy interior
            for hi in 0..h {
                for wi in 0..w {
                    padded[pad_base + (hi + 1) * pad_w + (wi + 1)] = x[in_base + hi * w + wi];
                }
            }
            // Reflect borders: top/bottom
            for wi in 0..w {
                padded[pad_base + 0 * pad_w + (wi + 1)] = x[in_base + 1 * w + wi];
                padded[pad_base + (pad_h - 1) * pad_w + (wi + 1)] = x[in_base + (h - 2) * w + wi];
            }
            // Reflect borders: left/right
            for hi in 0..h {
                padded[pad_base + (hi + 1) * pad_w + 0] = x[in_base + hi * w + 1];
                padded[pad_base + (hi + 1) * pad_w + (pad_w - 1)] = x[in_base + hi * w + (w - 2)];
            }
            // Reflect corners
            padded[pad_base + 0 * pad_w + 0] = x[in_base + 1 * w + 1];
            padded[pad_base + 0 * pad_w + (pad_w - 1)] = x[in_base + 1 * w + (w - 2)];
            padded[pad_base + (pad_h - 1) * pad_w + 0] = x[in_base + (h - 2) * w + 1];
            padded[pad_base + (pad_h - 1) * pad_w + (pad_w - 1)] =
                x[in_base + (h - 2) * w + (w - 2)];
        }
    }

    let out_h = (pad_h - 3) / stride + 1;
    let out_w = (pad_w - 3) / stride + 1;
    let mut out = vec![0.0f32; b * c * out_h * out_w];

    for bi in 0..b {
        for ci in 0..c {
            let pad_base = (bi * c + ci) * (pad_h * pad_w);
            let out_base = (bi * c + ci) * (out_h * out_w);
            let k_base = ci * 9;

            for oh in 0..out_h {
                let sh = oh * stride;
                for ow in 0..out_w {
                    let sw = ow * stride;
                    let mut sum_val = 0.0f32;
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let p_val = padded[pad_base + (sh + kh) * pad_w + (sw + kw)];
                            let k_val = kernel[k_base + kh * 3 + kw];
                            sum_val += p_val * k_val;
                        }
                    }
                    out[out_base + oh * out_w + ow] = sum_val;
                }
            }
        }
    }
    out
}

/// PixelUnshuffle forward on CPU: B x C x H x W -> B x (C * r^2) x (H / r) x (W / r).
pub fn pixel_unshuffle_cpu(
    x: &[f32],
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    r: usize,
) -> Vec<f32> {
    let out_h = h / r;
    let out_w = w / r;
    let out_c = c * r * r;
    let mut out = vec![0.0f32; b * out_c * out_h * out_w];

    for bi in 0..b {
        for ci in 0..c {
            for oh in 0..out_h {
                for rh in 0..r {
                    for ow in 0..out_w {
                        for rw in 0..r {
                            let in_idx =
                                bi * (c * h * w) + ci * (h * w) + (oh * r + rh) * w + (ow * r + rw);
                            let target_c = ci * (r * r) + rh * r + rw;
                            let out_idx = bi * (out_c * out_h * out_w)
                                + target_c * (out_h * out_w)
                                + oh * out_w
                                + ow;
                            out[out_idx] = x[in_idx];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Pooling by Multihead Attention (PMA) forward on CPU: Q = S_seed, K = X, V = X.
pub fn pma_cpu(
    x: &[f32],
    s_seed: &[f32], // [1 * 1 * dim]
    b: usize,
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0f32; b * dim];

    for bi in 0..b {
        let x_base = bi * (seq_len * dim);

        // 1. Compute dot-product scores: [seq_len]
        let mut scores = vec![0.0f32; seq_len];
        let mut max_score = f32::NEG_INFINITY;
        for s in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += s_seed[d] * x[x_base + s * dim + d];
            }
            let sc = dot * scale;
            scores[s] = sc;
            if sc > max_score {
                max_score = sc;
            }
        }

        // 2. Softmax weights
        let mut sum_exp = 0.0f32;
        let mut weights = vec![0.0f32; seq_len];
        for s in 0..seq_len {
            let exp_val = (scores[s] - max_score).exp();
            weights[s] = exp_val;
            sum_exp += exp_val;
        }
        for s in 0..seq_len {
            weights[s] /= sum_exp;
        }

        // 3. Aggregate values: sum(weight[s] * x[s, d])
        for d in 0..dim {
            let mut sum_d = 0.0f32;
            for s in 0..seq_len {
                sum_d += weights[s] * x[x_base + s * dim + d];
            }
            out[bi * dim + d] = sum_d;
        }
    }
    out
}

/// WGPU compute implementation for GeM Pooling.
pub async fn gem_pool_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &[f32],
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    p: f32,
    eps: f32,
) -> Option<Vec<f32>> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("GeM Shader"),
        source: wgpu::ShaderSource::Wgsl(GEM_SHADER.into()),
    });

    let total_channels = (b * c) as u32;
    let _in_bytes = std::mem::size_of_val(x) as u64;
    let out_bytes = ((b * c) * std::mem::size_of::<f32>()) as u64;

    let in_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GeM In Buffer"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GeM Out Buffer"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GeM Staging Buffer"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = GeMUniforms {
        b: b as u32,
        c: c as u32,
        h: h as u32,
        w: w as u32,
        p,
        eps,
        _pad0: 0,
        _pad1: 0,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GeM Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("GeM Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("GeM Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("GeM Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("GeM Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GeM Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GeM Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = total_channels.div_ceil(64);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging_buffer, 0, out_bytes);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = sender.send(v);
    });

    let _ = device.poll(wgpu::MaintainBase::Wait);
    receiver.recv().ok()?.ok()?;

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    Some(result)
}

/// Context-aware auto dispatcher for GeM Pooling.
pub async fn gem_pool_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    x: &[f32],
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    p: f32,
    eps: f32,
) -> Vec<f32> {
    if let Some(ctx) = context
        && let Some(res) = gem_pool_wgpu(&ctx.0, &ctx.1, x, b, c, h, w, p, eps).await
    {
        return res;
    }
    gem_pool_cpu(x, b, c, h, w, p, eps)
}
