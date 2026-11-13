//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 02:
//! 02_modern_sign_and_factored_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Fused Lion Step:
/// update = sign(beta1 * exp_avg + (1 - beta1) * grad)
/// param = param * (1 - lr * weight_decay) - lr * update
/// exp_avg = beta2 * exp_avg + (1 - beta2) * grad
pub const LION_SHADER: &str = r#"
struct LionUniforms {
    len: u32,
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> uniforms: LionUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read> m_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> param_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> m_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }

    let p = param_in[idx];
    let g = grad[idx];
    let m = m_in[idx];

    let interp = uniforms.beta1 * m + (1.0 - uniforms.beta1) * g;
    let update = sign(interp);

    let shrunk_p = p * (1.0 - uniforms.lr * uniforms.weight_decay);
    let new_p = shrunk_p - uniforms.lr * update;
    let new_m = uniforms.beta2 * m + (1.0 - uniforms.beta2) * g;

    param_out[idx] = new_p;
    m_out[idx] = new_m;
}
"#;

/// CPU reference routine for fused Lion step
pub fn lion_cpu(
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    out_param: &mut [f32],
    out_m: &mut [f32],
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
) {
    let n = param.len();
    assert_eq!(grad.len(), n);
    assert_eq!(m_buf.len(), n);
    assert_eq!(out_param.len(), n);
    assert_eq!(out_m.len(), n);

    for i in 0..n {
        let p = param[i];
        let g = grad[i];
        let m = m_buf[i];

        let interp = beta1 * m + (1.0 - beta1) * g;
        let update = if interp > 0.0 {
            1.0
        } else if interp < 0.0 {
            -1.0
        } else {
            0.0
        };

        let shrunk_p = p * (1.0 - lr * weight_decay);
        let new_p = shrunk_p - lr * update;
        let new_m = beta2 * m + (1.0 - beta2) * g;

        out_param[i] = new_p;
        out_m[i] = new_m;
    }
}

/// Auto-dispatcher executing fused Lion step on WGPU with CPU fallback
pub fn lion_auto(
    device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = param.len();
    let mut out_param = vec![0.0f32; n];
    let mut out_m = vec![0.0f32; n];

    #[cfg(feature = "wgpu")]
    if let KernelDevice::Wgpu(ctx) = device
        && let Ok(res) = lion_wgpu(ctx, param, grad, m_buf, lr, weight_decay, beta1, beta2)
    {
        return res;
    }

    lion_cpu(
        param,
        grad,
        m_buf,
        &mut out_param,
        &mut out_m,
        lr,
        weight_decay,
        beta1,
        beta2,
    );
    (out_param, out_m)
}

#[cfg(feature = "wgpu")]
fn lion_wgpu(
    ctx: &crate::context::WgpuContext,
    param: &[f32],
    grad: &[f32],
    m_buf: &[f32],
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    use wgpu::util::DeviceExt;

    let n = param.len();
    let size_bytes = std::mem::size_of_val(param) as u64;

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Uniforms {
        len: u32,
        lr: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        _pad0: f32,
        _pad1: f32,
        _pad2: f32,
    }

    let uniforms = Uniforms {
        len: n as u32,
        lr,
        weight_decay,
        beta1,
        beta2,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };

    let u_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lion_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let p_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lion_param_in"),
            contents: bytemuck::cast_slice(param),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let g_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lion_grad_in"),
            contents: bytemuck::cast_slice(grad),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let m_in_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lion_m_in"),
            contents: bytemuck::cast_slice(m_buf),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let p_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lion_param_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let m_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lion_m_out"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cs_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lion_shader"),
            source: wgpu::ShaderSource::Wgsl(LION_SHADER.into()),
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lion_layout"),
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

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lion_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lion_bind_group"),
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
                resource: p_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: m_out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lion_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lion_compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (n as u32).div_ceil(64);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    let p_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lion_p_stage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let m_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lion_m_stage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&p_out_buf, 0, &p_stage, 0, size_bytes);
    encoder.copy_buffer_to_buffer(&m_out_buf, 0, &m_stage, 0, size_bytes);

    ctx.queue.submit(Some(encoder.finish()));

    let (tx_p, rx_p) = std::sync::mpsc::channel();
    let (tx_m, rx_m) = std::sync::mpsc::channel();

    let p_slice = p_stage.slice(..);
    p_slice.map_async(wgpu::MapMode::Read, move |v| tx_p.send(v).unwrap());
    let m_slice = m_stage.slice(..);
    m_slice.map_async(wgpu::MapMode::Read, move |v| tx_m.send(v).unwrap());

    ctx.device.poll(wgpu::Maintain::Wait);

    rx_p.recv()??;
    rx_m.recv()??;

    let res_p = bytemuck::cast_slice::<u8, f32>(&p_slice.get_mapped_range()).to_vec();
    let res_m = bytemuck::cast_slice::<u8, f32>(&m_slice.get_mapped_range()).to_vec();

    p_stage.unmap();
    m_stage.unmap();

    Ok((res_p, res_m))
}
