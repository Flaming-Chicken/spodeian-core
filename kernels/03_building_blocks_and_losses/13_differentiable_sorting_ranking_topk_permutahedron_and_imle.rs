//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 13:
//! Differentiable Sorting, Ranking, Top-k, Permutahedron, and I-MLE.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// WGSL compute shader for Differentiable Soft Histogram:
/// For each batch element b and bin j, accumulates:
/// bin_count = sum_{i=0}^{N-1} [sigmoid((x[b, i] - lo_j)/tau) - sigmoid((x[b, i] - hi_j)/tau)]
pub const SOFT_HISTOGRAM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_x: array<f32>;
@group(0) @binding(1) var<storage, read> bin_centers: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_hist: array<f32>;

struct Uniforms {
    batch_size: u32,
    n_samples: u32,
    n_bins: u32,
    bin_width: f32,
    tau: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let j = global_id.y;
    if (b >= params.batch_size || j >= params.n_bins) {
        return;
    }
    let c = bin_centers[j];
    let lo = c - 0.5 * params.bin_width;
    let hi = c + 0.5 * params.bin_width;

    var count: f32 = 0.0;
    for (var i: u32 = 0u; i < params.n_samples; i = i + 1u) {
        let val = in_x[b * params.n_samples + i];
        let sig_lo = 1.0 / (1.0 + exp(-(val - lo) / params.tau));
        let sig_hi = 1.0 / (1.0 + exp(-(val - hi) / params.tau));
        count = count + (sig_lo - sig_hi);
    }
    out_hist[b * params.n_bins + j] = count;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HistogramUniforms {
    batch_size: u32,
    n_samples: u32,
    n_bins: u32,
    bin_width: f32,
    tau: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
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

/// SoftSort on CPU.
pub fn soft_sort_cpu(
    scores: &[f32],
    batch_size: usize,
    k: usize,
    tau: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut p_soft_out = vec![0.0f32; batch_size * k * k];
    let mut sorted_approx_out = vec![0.0f32; batch_size * k];

    for b in 0..batch_size {
        let s_slice = &scores[b * k..(b + 1) * k];
        let mut s_sorted = s_slice.to_vec();
        s_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        for i in 0..k {
            let s_sort_i = s_sorted[i];
            let mut row = vec![0.0f32; k];
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..k {
                let dist = (s_sort_i - s_slice[j]).abs();
                let logit = -dist / tau;
                row[j] = logit;
                if logit > max_val {
                    max_val = logit;
                }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..k {
                let val = (row[j] - max_val).exp();
                row[j] = val;
                sum_exp += val;
            }
            for j in 0..k {
                let prob = row[j] / sum_exp;
                p_soft_out[b * k * k + i * k + j] = prob;
                sorted_approx_out[b * k + i] += prob * s_slice[j];
            }
        }
    }
    (p_soft_out, sorted_approx_out)
}

/// NeuralSort on CPU.
pub fn neural_sort_cpu(scores: &[f32], batch_size: usize, k: usize, tau: f32) -> Vec<f32> {
    let mut p_neural_out = vec![0.0f32; batch_size * k * k];

    for b in 0..batch_size {
        let s_slice = &scores[b * k..(b + 1) * k];
        let mut l1_sum = vec![0.0f32; k];
        for i in 0..k {
            let mut sum_diff = 0.0f32;
            for j in 0..k {
                sum_diff += (s_slice[i] - s_slice[j]).abs();
            }
            l1_sum[i] = sum_diff;
        }

        for j in 0..k {
            let weight_j = (k + 1) as f32 - 2.0f32 * ((j + 1) as f32);
            let mut row = vec![0.0f32; k];
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..k {
                let cost_ji = (weight_j * s_slice[i] - l1_sum[i]) / tau;
                row[i] = cost_ji;
                if cost_ji > max_val {
                    max_val = cost_ji;
                }
            }
            let mut sum_exp = 0.0f32;
            for i in 0..k {
                let val = (row[i] - max_val).exp();
                row[i] = val;
                sum_exp += val;
            }
            for i in 0..k {
                p_neural_out[b * k * k + j * k + i] = row[i] / sum_exp;
            }
        }
    }
    p_neural_out
}

/// Differentiable Top-k on CPU.
pub fn diff_topk_cpu(
    scores: &[f32],
    batch_size: usize,
    n: usize,
    k_elements: usize,
    tau: f32,
    num_bisection: usize,
) -> Vec<f32> {
    let mut w_out = vec![0.0f32; batch_size * n];

    for b in 0..batch_size {
        let s_slice = &scores[b * n..(b + 1) * n];
        let min_s = s_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let max_s = s_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut lo = min_s - 5.0f32 * tau;
        let mut hi = max_s + 5.0f32 * tau;

        for _ in 0..num_bisection {
            let mid = 0.5f32 * (lo + hi);
            let mut sum_w = 0.0f32;
            for i in 0..n {
                let sig = 1.0f32 / (1.0f32 + (-(s_slice[i] - mid) / tau).exp());
                sum_w += sig;
            }
            if sum_w > (k_elements as f32) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let mid = 0.5f32 * (lo + hi);
        for i in 0..n {
            w_out[b * n + i] = 1.0f32 / (1.0f32 + (-(s_slice[i] - mid) / tau).exp());
        }
    }
    w_out
}

/// Soft histogram on CPU.
pub fn soft_histogram_cpu(
    x: &[f32],
    bin_centers: &[f32],
    bin_width: f32,
    tau: f32,
    batch_size: usize,
    n: usize,
    m: usize,
) -> Vec<f32> {
    let mut hist_out = vec![0.0f32; batch_size * m];

    for b in 0..batch_size {
        let mut sum_hist = 0.0f32;
        for j in 0..m {
            let c = bin_centers[j];
            let lo = c - 0.5f32 * bin_width;
            let hi = c + 0.5f32 * bin_width;

            let mut bin_count = 0.0f32;
            for i in 0..n {
                let val = x[b * n + i];
                let sig_lo = 1.0f32 / (1.0f32 + (-(val - lo) / tau).exp());
                let sig_hi = 1.0f32 / (1.0f32 + (-(val - hi) / tau).exp());
                bin_count += sig_lo - sig_hi;
            }
            hist_out[b * m + j] = bin_count;
            sum_hist += bin_count;
        }

        let denom = sum_hist + 1e-12f32;
        for j in 0..m {
            hist_out[b * m + j] /= denom;
        }
    }
    hist_out
}

/// Soft quantile on CPU.
pub fn soft_quantile_cpu(
    scores: &[f32],
    quantile: f32,
    tau: f32,
    batch_size: usize,
    n: usize,
) -> Vec<f32> {
    let target_rank = quantile * ((n - 1) as f32) + 1.0f32;
    let mut out = vec![0.0f32; batch_size];

    for b in 0..batch_size {
        let s_slice = &scores[b * n..(b + 1) * n];
        let mut ranks = vec![0.5f32; n];
        for i in 0..n {
            let mut r = 0.5f32;
            for j in 0..n {
                let diff = (s_slice[i] - s_slice[j]) / tau;
                r += 1.0f32 / (1.0f32 + (-diff).exp());
            }
            ranks[i] = r;
        }

        let mut weights = vec![0.0f32; n];
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..n {
            let dist_sq = (ranks[i] - target_rank) * (ranks[i] - target_rank);
            let logit = -dist_sq / (2.0f32 * tau);
            weights[i] = logit;
            if logit > max_val {
                max_val = logit;
            }
        }
        let mut sum_exp = 0.0f32;
        for i in 0..n {
            let val = (weights[i] - max_val).exp();
            weights[i] = val;
            sum_exp += val;
        }

        let mut q_val = 0.0f32;
        for i in 0..n {
            q_val += (weights[i] / sum_exp) * s_slice[i];
        }
        out[b] = q_val;
    }
    out
}

/// Dispatches soft histogram raw counts to WebGPU WGSL compute shader if context available.
pub fn soft_histogram_wgpu(
    context: &crate::WgpuContext,
    x: &[f32],
    bin_centers: &[f32],
    bin_width: f32,
    tau: f32,
    batch_size: usize,
    n_samples: usize,
    n_bins: usize,
) -> Option<Vec<f32>> {
    let device = &context.device;
    let queue = &context.queue;

    let total_elements = batch_size * n_bins;
    let output_size = (total_elements * std::mem::size_of::<f32>()) as u64;

    let buffer_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hist_in_x"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_centers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hist_in_centers"),
        contents: bytemuck::cast_slice(bin_centers),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hist_out"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hist_staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = HistogramUniforms {
        batch_size: batch_size as u32,
        n_samples: n_samples as u32,
        n_bins: n_bins as u32,
        bin_width,
        tau,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hist_uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hist_shader"),
        source: wgpu::ShaderSource::Wgsl(SOFT_HISTOGRAM_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hist_bgl"),
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hist_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_centers.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hist_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hist_compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hist_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hist_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (batch_size as u32).div_ceil(8);
        let workgroups_y = (n_bins as u32).div_ceil(8);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    encoder.copy_buffer_to_buffer(&buffer_out, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device.poll(wgpu::Maintain::Wait);

    if let Ok(Ok(())) = receiver.recv() {
        let data = buffer_slice.get_mapped_range();
        let raw_counts: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Normalize per batch
        let mut normalized = raw_counts;
        for b in 0..batch_size {
            let row = &mut normalized[b * n_bins..(b + 1) * n_bins];
            let sum: f32 = row.iter().sum::<f32>() + 1e-12f32;
            for val in row.iter_mut() {
                *val /= sum;
            }
        }
        Some(normalized)
    } else {
        None
    }
}

/// Auto-dispatcher: evaluates soft histogram using WebGPU WGSL if context available, else pure-CPU.
pub fn soft_histogram_auto(
    context: Option<&crate::WgpuContext>,
    x: &[f32],
    bin_centers: &[f32],
    bin_width: f32,
    tau: f32,
    batch_size: usize,
    n_samples: usize,
    n_bins: usize,
) -> Vec<f32> {
    if let Some(ctx) = context
        && let Some(res) = soft_histogram_wgpu(
            ctx,
            x,
            bin_centers,
            bin_width,
            tau,
            batch_size,
            n_samples,
            n_bins,
        )
    {
        return res;
    }
    soft_histogram_cpu(
        x,
        bin_centers,
        bin_width,
        tau,
        batch_size,
        n_samples,
        n_bins,
    )
}
