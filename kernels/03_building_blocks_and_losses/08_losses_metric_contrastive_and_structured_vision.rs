//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 08:
//! Losses: Metric, Contrastive, and Structured Vision.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL compute shader for elementwise reduction components of Soft-Dice Loss.
pub const DICE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_probs: array<f32>;
@group(0) @binding(1) var<storage, read> in_targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_inter: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_den: array<f32>;

struct Uniforms {
    total_elements: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    let p = in_probs[idx];
    let t = in_targets[idx];
    out_inter[idx] = p * t;
    out_den[idx] = p * p + t * t;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DiceUniforms {
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

/// Helper for L2 normalization.
fn l2_normalize_cpu(x: &[f32], b: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * d];
    for bi in 0..b {
        let base = bi * d;
        let mut sum_sq = 0.0f32;
        for j in 0..d {
            sum_sq += x[base + j].powi(2);
        }
        let norm = sum_sq.max(1e-12).sqrt();
        for j in 0..d {
            out[base + j] = x[base + j] / norm;
        }
    }
    out
}

/// InfoNCE Loss on CPU.
pub fn infonce_loss_cpu(q: &[f32], k: &[f32], b: usize, d: usize, tau: f32) -> f32 {
    let q_norm = l2_normalize_cpu(q, b, d);
    let k_norm = l2_normalize_cpu(k, b, d);

    let mut total_loss = 0.0f32;
    for i in 0..b {
        let mut max_val = f32::NEG_INFINITY;
        let mut row = vec![0.0f32; b];
        for j in 0..b {
            let mut dot = 0.0f32;
            for c in 0..d {
                dot += q_norm[i * d + c] * k_norm[j * d + c];
            }
            let logit = dot / tau;
            row[j] = logit;
            if logit > max_val {
                max_val = logit;
            }
        }

        let mut sum_exp = 0.0f32;
        for j in 0..b {
            sum_exp += (row[j] - max_val).exp();
        }
        let lse = max_val + sum_exp.ln();
        total_loss += -(row[i] - lse);
    }
    total_loss / (b as f32)
}

/// SigLIP Loss on CPU.
pub fn siglip_loss_cpu(q: &[f32], k: &[f32], b: usize, d: usize, scale: f32, bias: f32) -> f32 {
    let q_norm = l2_normalize_cpu(q, b, d);
    let k_norm = l2_normalize_cpu(k, b, d);

    let mut total_loss = 0.0f32;
    for i in 0..b {
        for j in 0..b {
            let mut dot = 0.0f32;
            for c in 0..d {
                dot += q_norm[i * d + c] * k_norm[j * d + c];
            }
            let logit = dot * scale + bias;
            let label = if i == j { 1.0f32 } else { -1.0f32 };
            let z = -label * logit;
            let max_val = z.max(0.0);
            let softplus = max_val + (1.0 + (-z.abs()).exp()).ln();
            total_loss += softplus;
        }
    }
    total_loss / ((b * b) as f32)
}

/// ArcFace Loss on CPU.
pub fn arcface_loss_cpu(
    embeddings: &[f32],
    weights: &[f32],
    targets: &[f32],
    b: usize,
    num_classes: usize,
    d: usize,
    s: f32,
    m: f32,
) -> f32 {
    let emb_norm = l2_normalize_cpu(embeddings, b, d);
    let w_norm = l2_normalize_cpu(weights, num_classes, d);

    let cos_m = m.cos();
    let sin_m = m.sin();

    let mut total_loss = 0.0f32;
    for i in 0..b {
        let target_idx = targets[i] as usize;
        let mut row_logits = vec![0.0f32; num_classes];
        let mut max_val = f32::NEG_INFINITY;

        for c in 0..num_classes {
            let mut dot = 0.0f32;
            for k in 0..d {
                dot += emb_norm[i * d + k] * w_norm[c * d + k];
            }
            let cos_t = dot.clamp(-0.9999999, 0.9999999);
            if c == target_idx {
                let sin_t = (1.0 - cos_t * cos_t).max(1e-7).sqrt();
                let cos_t_m = cos_t * cos_m - sin_t * sin_m;
                row_logits[c] = s * cos_t_m;
            } else {
                row_logits[c] = s * cos_t;
            }
            if row_logits[c] > max_val {
                max_val = row_logits[c];
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            sum_exp += (row_logits[c] - max_val).exp();
        }
        let lse = max_val + sum_exp.ln();
        total_loss += -(row_logits[target_idx] - lse);
    }
    total_loss / (b as f32)
}

/// Soft-Dice Loss on CPU.
pub fn soft_dice_loss_cpu(probs: &[f32], targets: &[f32], eps: f32) -> f32 {
    let mut intersection = 0.0f32;
    let mut den = 0.0f32;
    for (&p, &t) in probs.iter().zip(targets.iter()) {
        intersection += p * t;
        den += p * p + t * t;
    }
    let dice = (2.0 * intersection + eps) / (den + eps);
    1.0 - dice
}

/// GIoU Loss on CPU.
pub fn giou_loss_cpu(boxes1: &[f32], boxes2: &[f32], n: usize) -> f32 {
    let mut total_loss = 0.0f32;
    let eps = 1e-7f32;

    for i in 0..n {
        let b1 = &boxes1[i * 4..(i + 1) * 4];
        let b2 = &boxes2[i * 4..(i + 1) * 4];

        let x1_i = b1[0].max(b2[0]);
        let y1_i = b1[1].max(b2[1]);
        let x2_i = b1[2].min(b2[2]);
        let y2_i = b1[3].min(b2[3]);
        let inter_area = (x2_i - x1_i).max(0.0) * (y2_i - y1_i).max(0.0);

        let area1 = (b1[2] - b1[0]).max(0.0) * (b1[3] - b1[1]).max(0.0);
        let area2 = (b2[2] - b2[0]).max(0.0) * (b2[3] - b2[1]).max(0.0);
        let union_area = area1 + area2 - inter_area + eps;
        let iou = inter_area / union_area;

        let x1_c = b1[0].min(b2[0]);
        let y1_c = b1[1].min(b2[1]);
        let x2_c = b1[2].max(b2[2]);
        let y2_c = b1[3].max(b2[3]);
        let c_area = (x2_c - x1_c).max(0.0) * (y2_c - y1_c).max(0.0) + eps;

        let giou = iou - (c_area - union_area) / c_area;
        total_loss += 1.0 - giou;
    }
    total_loss / (n as f32)
}

/// WebGPU kernel execution for Soft-Dice Loss.
pub async fn dice_loss_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    probs: &[f32],
    targets: &[f32],
    eps: f32,
) -> Option<f32> {
    let total_elements = probs.len() as u32;
    let byte_size = std::mem::size_of_val(probs) as u64;

    let p_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dice Probs Buffer"),
        contents: bytemuck::cast_slice(probs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let t_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dice Targets Buffer"),
        contents: bytemuck::cast_slice(targets),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let inter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dice Inter Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let den_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dice Den Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniforms = DiceUniforms {
        total_elements,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dice Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging_inter = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dice Staging Inter"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_den = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dice Staging Den"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dice Shader"),
        source: wgpu::ShaderSource::Wgsl(DICE_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Dice Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
        label: Some("Dice Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Dice Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dice Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: t_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: inter_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: den_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Dice Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Dice Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = total_elements.div_ceil(64);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&inter_buffer, 0, &staging_inter, 0, byte_size);
    encoder.copy_buffer_to_buffer(&den_buffer, 0, &staging_den, 0, byte_size);
    queue.submit(Some(encoder.finish()));

    let slice_inter = staging_inter.slice(..);
    let slice_den = staging_den.slice(..);

    let (sender1, receiver1) = std::sync::mpsc::channel();
    let (sender2, receiver2) = std::sync::mpsc::channel();

    slice_inter.map_async(wgpu::MapMode::Read, move |v| {
        let _ = sender1.send(v);
    });
    slice_den.map_async(wgpu::MapMode::Read, move |v| {
        let _ = sender2.send(v);
    });

    let _ = device.poll(wgpu::MaintainBase::Wait);
    receiver1.recv().ok()?.ok()?;
    receiver2.recv().ok()?.ok()?;

    let data_inter = slice_inter.get_mapped_range();
    let inter_vec: Vec<f32> = bytemuck::cast_slice(&data_inter).to_vec();
    drop(data_inter);
    staging_inter.unmap();

    let data_den = slice_den.get_mapped_range();
    let den_vec: Vec<f32> = bytemuck::cast_slice(&data_den).to_vec();
    drop(data_den);
    staging_den.unmap();

    let intersection: f32 = inter_vec.iter().sum();
    let denominator: f32 = den_vec.iter().sum();
    let dice = (2.0 * intersection + eps) / (denominator + eps);
    Some(1.0 - dice)
}

/// Context-aware auto dispatcher for Soft-Dice Loss.
pub async fn soft_dice_auto(
    context: Option<Arc<(wgpu::Device, wgpu::Queue)>>,
    probs: &[f32],
    targets: &[f32],
    eps: f32,
) -> f32 {
    if let Some(ctx) = context
        && let Some(res) = dice_loss_wgpu(&ctx.0, &ctx.1, probs, targets, eps).await
    {
        return res;
    }
    soft_dice_loss_cpu(probs, targets, eps)
}
