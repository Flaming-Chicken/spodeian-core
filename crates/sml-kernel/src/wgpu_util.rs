//! Reusable utilities for compiling and dispatching WebGPU WGSL compute shaders.

#[cfg(feature = "wgpu")]
use crate::context::WgpuContext;
#[cfg(feature = "wgpu")]
use std::collections::HashMap;
#[cfg(feature = "wgpu")]
use std::sync::Arc;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "wgpu")]
use wgpu::util::DeviceExt;

#[cfg(feature = "wgpu")]
type PipelineCacheKey = (String, String, usize, bool);

#[cfg(feature = "wgpu")]
struct CachedPipeline {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

#[cfg(feature = "wgpu")]
type PipelineCache = HashMap<PipelineCacheKey, CachedPipeline>;

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
static PIPELINE_CACHE: OnceLock<Mutex<PipelineCache>> = OnceLock::new();

/// Runs `f` with the process-wide pipeline cache.
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn with_pipeline_cache<R>(f: impl FnOnce(&mut PipelineCache) -> R) -> R {
    let cache = PIPELINE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    f(&mut cache.lock().unwrap())
}

// wgpu handles aren't Send/Sync on wasm32, so the cache is per thread there.
#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
thread_local! {
    static PIPELINE_CACHE: std::cell::RefCell<PipelineCache> = std::cell::RefCell::new(HashMap::new());
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
fn with_pipeline_cache<R>(f: impl FnOnce(&mut PipelineCache) -> R) -> R {
    PIPELINE_CACHE.with(|cache| f(&mut cache.borrow_mut()))
}

#[cfg(feature = "wgpu")]
fn get_or_create_pipeline(
    device: &wgpu::Device,
    shader_src: &str,
    entry_point: &str,
    num_inputs: usize,
    has_uniform: bool,
) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
    let key = (
        shader_src.to_string(),
        entry_point.to_string(),
        num_inputs,
        has_uniform,
    );

    if let Some(hit) = with_pipeline_cache(|cache| {
        cache.get(&key).map(|cached| {
            (
                Arc::clone(&cached.pipeline),
                Arc::clone(&cached.bind_group_layout),
            )
        })
    }) {
        return hit;
    }

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cached_compute_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let mut bind_group_layout_entries = Vec::new();
    let mut binding_idx = 0u32;

    for _ in 0..num_inputs {
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding_idx,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        binding_idx += 1;
    }

    // Output storage buffer
    bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
        binding: binding_idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    binding_idx += 1;

    if has_uniform {
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding_idx,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    let bind_group_layout = Arc::new(device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bind_group_layout"),
            entries: &bind_group_layout_entries,
        },
    ));

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = Arc::new(device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        },
    ));

    with_pipeline_cache(|cache| {
        cache.insert(
            key,
            CachedPipeline {
                pipeline: Arc::clone(&compute_pipeline),
                bind_group_layout: Arc::clone(&bind_group_layout),
            },
        )
    });

    (compute_pipeline, bind_group_layout)
}

/// Executes a WGSL compute shader with input storage buffers, optional uniform buffer,
/// and returns the raw output bytes from the output storage buffer.
/// Caches compiled compute pipelines for instantaneous zero-overhead re-dispatch.
#[cfg(feature = "wgpu")]
pub fn run_compute_shader(
    ctx: &WgpuContext,
    shader_src: &str,
    entry_point: &str,
    inputs: &[&[u8]],
    output_len_bytes: usize,
    uniform_data: Option<&[u8]>,
    workgroups: (u32, u32, u32),
) -> Option<Vec<u8>> {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let (compute_pipeline, bind_group_layout) = get_or_create_pipeline(
        device,
        shader_src,
        entry_point,
        inputs.len(),
        uniform_data.is_some(),
    );

    // 1. Create input storage buffers
    let input_buffers: Vec<wgpu::Buffer> = inputs
        .iter()
        .map(|&inp| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input_storage_buffer"),
                contents: inp,
                usage: wgpu::BufferUsages::STORAGE,
            })
        })
        .collect();

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_storage_buffer"),
        size: output_len_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniform_buffer = uniform_data.map(|uniforms| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform_buffer"),
            contents: uniforms,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    });

    // 2. Construct bind group using cached layout
    let mut bind_group_entries = Vec::new();
    let mut binding_idx = 0u32;

    for buf in &input_buffers {
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: binding_idx,
            resource: buf.as_entire_binding(),
        });
        binding_idx += 1;
    }

    bind_group_entries.push(wgpu::BindGroupEntry {
        binding: binding_idx,
        resource: output_buffer.as_entire_binding(),
    });
    binding_idx += 1;

    if let Some(ref u_buf) = uniform_buffer {
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: binding_idx,
            resource: u_buf.as_entire_binding(),
        });
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    // 3. Staging Buffer for reading back results
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: output_len_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        output_len_bytes as u64,
    );
    queue.submit(Some(encoder.finish()));

    // 4. Map staging buffer back to host CPU memory
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = sender.send(v);
    });

    device.poll(wgpu::Maintain::Wait);

    if let Ok(Ok(())) = receiver.recv() {
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        Some(result)
    } else {
        None
    }
}
