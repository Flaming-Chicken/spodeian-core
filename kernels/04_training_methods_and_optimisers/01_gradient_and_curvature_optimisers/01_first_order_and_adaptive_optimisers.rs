//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 01:
//! 01_first_order_and_adaptive_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Fused AdamW Step:
/// Decoupled weight decay: p = p * (1 - lr * wd) - lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
pub const ADAMW_SHADER: &str = r#"
struct AdamWUniforms {
    len: u32,
    step: u32,
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: AdamWUniforms;
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

    let update = m_hat / (sqrt(v_hat) + uniforms.eps);
    let shrunk_p = p * (1.0 - uniforms.lr * uniforms.weight_decay);
    let new_p = shrunk_p - uniforms.lr * update;

    param_out[idx] = new_p;
    m_out[idx] = new_m;
    v_out[idx] = new_v;
}
"#;

/// CPU reference routine for fused AdamW step
pub fn adamw_cpu(
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    v_buf: &[f32],
    out_param: &mut [f32],
    out_m: &mut [f32],
    out_v: &mut [f32],
    step: u32,
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) {
    let n = param.len();
    assert_eq!(grad.len(), n);
    assert_eq!(m_buf.len(), n);
    assert_eq!(v_buf.len(), n);
    assert_eq!(out_param.len(), n);
    assert_eq!(out_m.len(), n);
    assert_eq!(out_v.len(), n);

    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);

    for i in 0..n {
        let p = param[i];
        let g = grad[i];
        let m = m_buf[i];
        let v = v_buf[i];

        let new_m = beta1 * m + (1.0 - beta1) * g;
        let new_v = beta2 * v + (1.0 - beta2) * (g * g);

        let m_hat = new_m / bc1;
        let v_hat = new_v / bc2;

        let update = m_hat / (v_hat.sqrt() + eps);
        let shrunk_p = p * (1.0 - lr * weight_decay);
        let new_p = shrunk_p - lr * update;

        out_param[i] = new_p;
        out_m[i] = new_m;
        out_v[i] = new_v;
    }
}

/// Auto-dispatcher executing fused AdamW step on WGPU with CPU fallback
pub fn adamw_auto(
    device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    v_buf: &[f32],
    step: u32,
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = param.len();
    let mut out_param = vec![0.0f32; n];
    let mut out_m = vec![0.0f32; n];
    let mut out_v = vec![0.0f32; n];

    #[cfg(feature = "wgpu")]
    if let KernelDevice::Wgpu(ctx) = device {
        if let Ok(res) = adamw_wgpu(
            ctx,
            param,
            grad,
            m_buf,
            v_buf,
            step,
            lr,
            weight_decay,
            beta1,
            beta2,
            eps,
        ) {
            return res;
        }
    }

    adamw_cpu(
        param,
        grad,
        m_buf,
        v_buf,
        &mut out_param,
        &mut out_m,
        &mut out_v,
        step,
        lr,
        weight_decay,
        beta1,
        beta2,
        eps,
    );
    (out_param, out_m, out_v)
}

#[cfg(feature = "wgpu")]
fn adamw_wgpu(
    ctx: &crate::context::WgpuContext,
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    v_buf: &[f32],
    step: u32,
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    use wgpu::util::DeviceExt;

    let n = param.len();
    let size_bytes = (n * std::mem::size_of::<f32>()) as u64;

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
        _pad: f32,
    }

    let uniforms = Uniforms {
        len: n as u32,
        step,
        lr,
        weight_decay,
        beta1,
        beta2,
        eps,
        _pad: 0.0,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("adamw_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let p_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("adamw_param_in"),
            contents: bytemuck::cast_slice(param),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let g_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("adamw_grad_in"),
            contents: bytemuck::cast_slice(grad),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let m_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("adamw_m_in"),
            contents: bytemuck::cast_slice(m_buf),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let v_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("adamw_v_in"),
            contents: bytemuck::cast_slice(v_buf),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let p_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adamw_param_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let m_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adamw_m_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let v_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adamw_v_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cs_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adamw_shader"),
            source: wgpu::ShaderSource::Wgsl(ADAMW_SHADER.into()),
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("adamw_layout"),
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
            label: Some("adamw_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("adamw_pipeline"),
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("adamw_bind_group"),
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
            label: Some("adamw_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("adamw_compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((n as u32) + 63) / 64;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    let p_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adamw_p_stage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let m_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adamw_m_stage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let v_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adamw_v_stage"),
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
