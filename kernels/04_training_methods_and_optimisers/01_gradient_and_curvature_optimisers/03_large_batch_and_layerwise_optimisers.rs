//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 03:
//! 03_large_batch_and_layerwise_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Fused LARS Step:
/// local_lr = lr * trust_ratio
/// total_g = grad + weight_decay * param
/// v = momentum * v + local_lr * total_g
/// param = param - v
pub const LARS_SHADER: &str = r#"
struct LarsUniforms {
    len: u32,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    trust_ratio: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> uniforms: LarsUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read> v_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> param_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> v_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }

    let p = param_in[idx];
    let g = grad[idx];
    let v = v_in[idx];

    let local_lr = uniforms.lr * uniforms.trust_ratio;
    let total_g = g + uniforms.weight_decay * p;
    let new_v = uniforms.momentum * v + local_lr * total_g;
    let new_p = p - new_v;

    param_out[idx] = new_p;
    v_out[idx] = new_v;
}
"#;

/// WGSL Compute Shader for Fused LAMB Step:
/// m = beta1 * m + (1 - beta1) * grad
/// v = beta2 * v + (1 - beta2) * grad^2
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// r = m_hat / (sqrt(v_hat) + eps) + weight_decay * param
/// param = param - lr * trust_ratio * r
pub const LAMB_SHADER: &str = r#"
struct LambUniforms {
    len: u32,
    step: u32,
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    trust_ratio: f32,
};

@group(0) @binding(0) var<uniform> uniforms: LambUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read> m_in: array<f32>;
@group(0) @binding(4) var<storage, read> v_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> param_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> m_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> v_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }

    let p = param_in[idx];
    let g = grad[idx];
    let m = m_in[idx];
    let v = v_in[idx];

    let new_m = uniforms.beta1 * m + (1.0 - uniforms.beta1) * g;
    let new_v = uniforms.beta2 * v + (1.0 - uniforms.beta2) * (g * g);

    let bc1 = 1.0 - pow(uniforms.beta1, f32(uniforms.step));
    let bc2 = 1.0 - pow(uniforms.beta2, f32(uniforms.step));

    let m_hat = new_m / bc1;
    let v_hat = new_v / bc2;

    let adam_step = m_hat / (sqrt(v_hat) + uniforms.eps);
    let r = adam_step + uniforms.weight_decay * p;
    let new_p = p - uniforms.lr * uniforms.trust_ratio * r;

    param_out[idx] = new_p;
    m_out[idx] = new_m;
    v_out[idx] = new_v;
}
"#;

/// CPU reference routine for LARS step.
pub fn lars_cpu(
    param: &[f32],
    grad: &[f32],
    v_buf: &[f32],
    out_param: &mut [f32],
    out_v: &mut [f32],
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    trust_coef: f32,
    eps: f32,
) -> f32 {
    let n = param.len();
    assert_eq!(grad.len(), n);
    assert_eq!(v_buf.len(), n);
    assert_eq!(out_param.len(), n);
    assert_eq!(out_v.len(), n);

    let mut w_sq = 0.0f32;
    let mut g_sq = 0.0f32;
    for i in 0..n {
        w_sq += param[i] * param[i];
        g_sq += grad[i] * grad[i];
    }
    let w_norm = w_sq.sqrt();
    let g_norm = g_sq.sqrt();

    let trust_ratio = if w_norm > 0.0 && g_norm > 0.0 {
        trust_coef * w_norm / (g_norm + weight_decay * w_norm + eps)
    } else {
        1.0
    };

    let local_lr = lr * trust_ratio;
    for i in 0..n {
        let total_g = grad[i] + weight_decay * param[i];
        let new_v = momentum * v_buf[i] + local_lr * total_g;
        out_v[i] = new_v;
        out_param[i] = param[i] - new_v;
    }

    trust_ratio
}

/// Auto-dispatcher executing LARS step with CPU fallback
pub fn lars_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    v_buf: &[f32],
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    trust_coef: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, f32) {
    let n = param.len();
    let mut out_param = vec![0.0f32; n];
    let mut out_v = vec![0.0f32; n];
    let ratio = lars_cpu(
        param,
        grad,
        v_buf,
        &mut out_param,
        &mut out_v,
        lr,
        momentum,
        weight_decay,
        trust_coef,
        eps,
    );
    (out_param, out_v, ratio)
}

/// CPU reference routine for LAMB step.
pub fn lamb_cpu(
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    v_buf: &[f32],
    out_param: &mut [f32],
    out_m: &mut [f32],
    out_v: &mut [f32],
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) -> f32 {
    let n = param.len();
    assert_eq!(grad.len(), n);
    assert_eq!(m_buf.len(), n);
    assert_eq!(v_buf.len(), n);
    assert_eq!(out_param.len(), n);
    assert_eq!(out_m.len(), n);
    assert_eq!(out_v.len(), n);

    let bc1 = (1.0 - (beta1 as f64).powi(step as i32)) as f32;
    let bc2 = (1.0 - (beta2 as f64).powi(step as i32)) as f32;

    let mut w_sq = 0.0f32;
    let mut r_sq = 0.0f32;
    let mut r_buf = vec![0.0f32; n];

    for i in 0..n {
        let new_m = beta1 * m_buf[i] + (1.0 - beta1) * grad[i];
        let new_v = beta2 * v_buf[i] + (1.0 - beta2) * (grad[i] * grad[i]);
        out_m[i] = new_m;
        out_v[i] = new_v;

        let m_hat = new_m / bc1;
        let v_hat = new_v / bc2;
        let adam_step = m_hat / (v_hat.sqrt() + eps);
        let r = adam_step + weight_decay * param[i];
        r_buf[i] = r;

        w_sq += param[i] * param[i];
        r_sq += r * r;
    }

    let w_norm = w_sq.sqrt();
    let r_norm = r_sq.sqrt();
    let trust_ratio = if w_norm > 0.0 && r_norm > 0.0 {
        w_norm / r_norm
    } else {
        1.0
    };

    let step_scale = lr * trust_ratio;
    for i in 0..n {
        out_param[i] = param[i] - step_scale * r_buf[i];
    }

    trust_ratio
}

/// Auto-dispatcher executing LAMB step with WGPU compute shader and CPU fallback.
pub fn lamb_auto(
    device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    v_buf: &[f32],
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let n = param.len();

    #[cfg(feature = "wgpu")]
    if let KernelDevice::Wgpu(ctx) = device {
        // 1. Compute trust ratio
        let bc1 = (1.0 - (beta1 as f64).powi(step as i32)) as f32;
        let bc2 = (1.0 - (beta2 as f64).powi(step as i32)) as f32;
        let mut w_sq = 0.0f32;
        let mut r_sq = 0.0f32;

        for i in 0..n {
            let nm = beta1 * m_buf[i] + (1.0 - beta1) * grad[i];
            let nv = beta2 * v_buf[i] + (1.0 - beta2) * (grad[i] * grad[i]);
            let m_hat = nm / bc1;
            let v_hat = nv / bc2;
            let r = (m_hat / (v_hat.sqrt() + eps)) + weight_decay * param[i];
            w_sq += param[i] * param[i];
            r_sq += r * r;
        }

        let w_norm = w_sq.sqrt();
        let r_norm = r_sq.sqrt();
        let trust_ratio = if w_norm > 0.0 && r_norm > 0.0 {
            w_norm / r_norm
        } else {
            1.0
        };

        if let Ok((p, m, v)) = run_lamb_wgpu(
            ctx,
            param,
            grad,
            m_buf,
            v_buf,
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            trust_ratio,
        ) {
            return (p, m, v, trust_ratio);
        }
    }

    let mut out_param = vec![0.0f32; n];
    let mut out_m = vec![0.0f32; n];
    let mut out_v = vec![0.0f32; n];
    let ratio = lamb_cpu(
        param,
        grad,
        m_buf,
        v_buf,
        &mut out_param,
        &mut out_m,
        &mut out_v,
        step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
    );
    (out_param, out_m, out_v, ratio)
}

#[cfg(feature = "wgpu")]
fn run_lamb_wgpu(
    ctx: &crate::context::WgpuContext,
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    v_buf: &[f32],
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    trust_ratio: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    use wgpu::util::DeviceExt;

    let n = param.len();
    let size_bytes = std::mem::size_of_val(param) as u64;

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Uniforms {
        len: u32,
        step: u32,
        lr: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        trust_ratio: f32,
    }

    let uniforms = Uniforms {
        len: n as u32,
        step,
        lr,
        weight_decay,
        beta1,
        beta2,
        eps,
        trust_ratio,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lamb_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let p_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lamb_p_in"),
            contents: bytemuck::cast_slice(param),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let g_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lamb_g_in"),
            contents: bytemuck::cast_slice(grad),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let m_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lamb_m_in"),
            contents: bytemuck::cast_slice(m_buf),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let v_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lamb_v_in"),
            contents: bytemuck::cast_slice(v_buf),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let p_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lamb_p_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let m_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lamb_m_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let v_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lamb_v_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cs_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lamb_shader"),
            source: wgpu::ShaderSource::Wgsl(LAMB_SHADER.into()),
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lamb_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lamb_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lamb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lamb_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: u_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: p_in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: g_in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: m_in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: v_in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: p_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: m_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: v_out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lamb_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lamb_compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (n as u32).div_ceil(64);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    let p_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lamb_p_stage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let m_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lamb_m_stage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let v_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lamb_v_stage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&p_out_buf, 0, &p_stage, 0, size_bytes);
    encoder.copy_buffer_to_buffer(&m_out_buf, 0, &m_stage, 0, size_bytes);
    encoder.copy_buffer_to_buffer(&v_out_buf, 0, &v_stage, 0, size_bytes);

    ctx.queue.submit(Some(encoder.finish()));

    let (tx_p, rx_p) = std::sync::mpsc::channel();
    let (tx_m, rx_m) = std::sync::mpsc::channel();
    let (tx_v, rx_v) = std::sync::mpsc::channel();

    let p_slice = p_stage.slice(..);
    p_slice.map_async(wgpu::MapMode::Read, move |v| tx_p.send(v).unwrap());
    let m_slice = m_stage.slice(..);
    m_slice.map_async(wgpu::MapMode::Read, move |v| tx_m.send(v).unwrap());
    let v_slice = v_stage.slice(..);
    v_slice.map_async(wgpu::MapMode::Read, move |v| tx_v.send(v).unwrap());

    ctx.device.poll(wgpu::Maintain::Wait);

    rx_p.recv()??;
    rx_m.recv()??;
    rx_v.recv()??;

    let res_p = bytemuck::cast_slice::<u8, f32>(&p_slice.get_mapped_range()).to_vec();
    let res_m = bytemuck::cast_slice::<u8, f32>(&m_slice.get_mapped_range()).to_vec();
    let res_v = bytemuck::cast_slice::<u8, f32>(&v_slice.get_mapped_range()).to_vec();

    p_stage.unmap();
    m_stage.unmap();
    v_stage.unmap();

    Ok((res_p, res_m, res_v))
}

/// CPU reference routine for Gradient Noise Scale (B_simple = Tr(Sigma) / ||G||^2).
pub fn gradient_noise_scale_cpu(
    per_sample_grads: &[f32],
    b_size: usize,
    dim: usize,
) -> (f32, f32, f32) {
    assert_eq!(per_sample_grads.len(), b_size * dim);
    let mut g_mean = vec![0.0f32; dim];
    for b in 0..b_size {
        for d in 0..dim {
            g_mean[d] += per_sample_grads[b * dim + d];
        }
    }
    for d in 0..dim {
        g_mean[d] /= b_size as f32;
    }

    let mut g_norm_sq = 0.0f32;
    for d in 0..dim {
        g_norm_sq += g_mean[d] * g_mean[d];
    }

    let mut sum_sq_diff = 0.0f32;
    for b in 0..b_size {
        for d in 0..dim {
            let diff = per_sample_grads[b * dim + d] - g_mean[d];
            sum_sq_diff += diff * diff;
        }
    }
    let trace_sigma = sum_sq_diff / ((b_size - 1) as f32);
    let b_simple = trace_sigma / (g_norm_sq + 1e-12);

    (g_norm_sq, trace_sigma, b_simple)
}
