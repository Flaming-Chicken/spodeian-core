//! Dual-engine WebGPU + CPU Hardware Kernels for Category 03 Topic 12:
//! Latent Decision Gates: Highway Networks, Hard Concrete L0, Sparsemax, PonderNet Halting, and FiLM.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// WGSL compute shader for Feature-wise Linear Modulation (FiLM):
/// out[b * dim + d] = gamma[b * dim + d] * x[b * dim + d] + beta[b * dim + d]
pub const FILM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> in_x: array<f32>;
@group(0) @binding(1) var<storage, read> in_gamma: array<f32>;
@group(0) @binding(2) var<storage, read> in_beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_y: array<f32>;

struct Uniforms {
    batch_size: u32,
    dim: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let d = global_id.y;
    if (b >= params.batch_size || d >= params.dim) {
        return;
    }
    let idx = b * params.dim + d;
    out_y[idx] = in_gamma[idx] * in_x[idx] + in_beta[idx];
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FilmUniforms {
    batch_size: u32,
    dim: u32,
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

/// Highway network forward on CPU.
pub fn highway_forward_cpu(
    x: &[f32],
    w_h: &[f32],
    b_h: &[f32],
    w_t: &[f32],
    b_t: &[f32],
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * dim];
    for b in 0..batch_size {
        for j in 0..dim {
            let mut sum_h = b_h[j];
            let mut sum_t = b_t[j];
            for i in 0..dim {
                let x_val = x[b * dim + i];
                sum_h += x_val * w_h[i * dim + j];
                sum_t += x_val * w_t[i * dim + j];
            }
            let h = sum_h.max(0.0f32);
            let t = 1.0f32 / (1.0f32 + (-sum_t).exp());
            out[b * dim + j] = t * h + (1.0f32 - t) * x[b * dim + j];
        }
    }
    out
}

/// Gated Linear Unit (GLU) on CPU.
pub fn glu_forward_cpu(x: &[f32], w: &[f32], b: &[f32], batch_size: usize, dim: usize) -> Vec<f32> {
    let d2 = 2 * dim;
    let mut z = vec![0.0f32; batch_size * d2];
    for b_idx in 0..batch_size {
        for j in 0..d2 {
            let mut sum = b[j];
            for i in 0..dim {
                sum += x[b_idx * dim + i] * w[i * d2 + j];
            }
            z[b_idx * d2 + j] = sum;
        }
    }

    let mut out = vec![0.0f32; batch_size * dim];
    for b_idx in 0..batch_size {
        for j in 0..dim {
            let a = z[b_idx * d2 + j];
            let gate = z[b_idx * d2 + dim + j];
            let sig = 1.0f32 / (1.0f32 + (-gate).exp());
            out[b_idx * dim + j] = a * sig;
        }
    }
    out
}

/// Hard concrete gate forward on CPU.
pub fn hard_concrete_cpu(
    log_alpha: &[f32],
    u: &[f32],
    batch_size: usize,
    dim: usize,
    beta: f32,
    gamma: f32,
    zeta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let total = batch_size * dim;
    let mut g_out = vec![0.0f32; total];
    let mut l0_out = vec![0.0f32; total];

    let offset = beta * (-gamma / zeta).ln();
    for i in 0..total {
        let u_val = u[i];
        let logit_u = u_val.ln() - (1.0f32 - u_val).ln();
        let la_val = log_alpha[i];
        let s = 1.0f32 / (1.0f32 + (-(logit_u + la_val) / beta).exp());
        let s_bar = s * (zeta - gamma) + gamma;
        g_out[i] = s_bar.clamp(0.0f32, 1.0f32);

        let l0_prob = 1.0f32 / (1.0f32 + (-(la_val - offset)).exp());
        l0_out[i] = l0_prob;
    }
    (g_out, l0_out)
}

/// Sparsemax on CPU.
pub fn sparsemax_cpu(z: &[f32], batch_size: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * dim];
    for b in 0..batch_size {
        let row = &z[b * dim..(b + 1) * dim];
        let mut sorted = row.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = vec![0.0f32; dim];
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += sorted[j];
            cumsum[j] = sum;
        }

        let mut k_star = 1usize;
        for j in 0..dim {
            let bound = 1.0f32 + ((j + 1) as f32) * sorted[j];
            if bound > cumsum[j] {
                k_star = j + 1;
            }
        }

        let tau = (cumsum[k_star - 1] - 1.0f32) / (k_star as f32);
        for j in 0..dim {
            out[b * dim + j] = (row[j] - tau).max(0.0f32);
        }
    }
    out
}

/// PonderNet halting on CPU.
pub fn pondernet_halting_cpu(
    lambdas: &[f32],
    batch_size: usize,
    n_steps: usize,
    lambda_p: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut p_out = vec![0.0f32; batch_size * n_steps];
    let mut exp_steps_out = vec![0.0f32; batch_size];
    let mut kl_out = vec![0.0f32; batch_size];

    let mut p_star = vec![0.0f32; n_steps];
    for s in 1..=n_steps {
        p_star[s - 1] = lambda_p * (1.0f32 - lambda_p).powi((s - 1) as i32);
    }

    for b in 0..batch_size {
        let mut cum_unhalted = 1.0f32;
        let mut steps_acc = 0.0f32;
        let mut kl_acc = 0.0f32;

        for s in 0..n_steps {
            let lam = lambdas[b * n_steps + s];
            let p_step = lam * cum_unhalted;
            p_out[b * n_steps + s] = p_step;
            cum_unhalted *= 1.0f32 - lam;

            let step_num = (s + 1) as f32;
            steps_acc += p_step * step_num;

            let p_clamped = p_step.max(eps);
            let p_star_clamped = p_star[s].max(eps);
            kl_acc += p_step * (p_clamped.ln() - p_star_clamped.ln());
        }

        exp_steps_out[b] = steps_acc;
        kl_out[b] = kl_acc;
    }
    (p_out, exp_steps_out, kl_out)
}

/// FiLM on CPU.
pub fn film_cpu(x: &[f32], gamma: &[f32], beta: &[f32], batch_size: usize, dim: usize) -> Vec<f32> {
    let total = batch_size * dim;
    let mut out = vec![0.0f32; total];
    for i in 0..total {
        out[i] = gamma[i] * x[i] + beta[i];
    }
    out
}

/// Dispatches FiLM to WebGPU WGSL compute shader if context available.
pub fn film_wgpu(
    context: &crate::WgpuContext,
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    dim: usize,
) -> Option<Vec<f32>> {
    let device = &context.device;
    let queue = &context.queue;

    let total_elements = batch_size * dim;
    let output_size = (total_elements * std::mem::size_of::<f32>()) as u64;

    let buffer_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("film_in_x"),
        contents: bytemuck::cast_slice(x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_gamma = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("film_in_gamma"),
        contents: bytemuck::cast_slice(gamma),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_beta = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("film_in_beta"),
        contents: bytemuck::cast_slice(beta),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("film_out"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("film_staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = FilmUniforms {
        batch_size: batch_size as u32,
        dim: dim as u32,
        _pad1: 0,
        _pad2: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("film_uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("film_shader"),
        source: wgpu::ShaderSource::Wgsl(FILM_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("film_bgl"),
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("film_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_gamma.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_beta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("film_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("film_compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("film_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("film_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (batch_size as u32).div_ceil(8);
        let workgroups_y = (dim as u32).div_ceil(8);
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
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Some(result)
    } else {
        None
    }
}

/// Auto-dispatcher: evaluates FiLM using WebGPU WGSL if context available, else pure-CPU.
pub fn film_auto(
    context: Option<&crate::WgpuContext>,
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    if let Some(ctx) = context
        && let Some(res) = film_wgpu(ctx, x, gamma, beta, batch_size, dim)
    {
        return res;
    }
    film_cpu(x, gamma, beta, batch_size, dim)
}
