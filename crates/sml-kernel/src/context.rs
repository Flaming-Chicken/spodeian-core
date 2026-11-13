//! Device and context management for CPU and WebGPU backends.

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

/// Cached singleton WebGPU context across threads/tests to prevent graphics driver lock contention.
#[cfg(not(target_arch = "wasm32"))]
static GLOBAL_WGPU_CONTEXT: OnceLock<Option<WgpuContext>> = OnceLock::new();

/// WebGPU device and queue context.
#[derive(Debug, Clone)]
pub struct WgpuContext {
    #[cfg(feature = "wgpu")]
    pub device: Arc<wgpu::Device>,
    #[cfg(feature = "wgpu")]
    pub queue: Arc<wgpu::Queue>,
}

impl WgpuContext {
    /// Returns the global singleton WebGPU context, initializing it once if available.
    ///
    /// On wasm32 this is always `None` (CPU fallback): device setup blocks on
    /// `pollster`, which can't drive the browser's WebGPU futures, and wgpu
    /// handles there are neither `Send` nor `Sync`.
    pub fn global() -> Option<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        return GLOBAL_WGPU_CONTEXT.get_or_init(Self::new).clone();
        #[cfg(target_arch = "wasm32")]
        None
    }

    #[cfg(feature = "wgpu")]
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ai-kernel-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    #[cfg(not(feature = "wgpu"))]
    pub fn new() -> Option<Self> {
        None
    }
}

/// Execution device abstraction: CPU or WebGPU with automatic fallback.
#[derive(Debug, Clone)]
pub enum KernelDevice {
    Cpu,
    Wgpu(WgpuContext),
}

impl KernelDevice {
    /// Attempts to initialize WebGPU; gracefully falls back to CPU if unavailable.
    pub fn auto() -> Self {
        #[cfg(feature = "wgpu")]
        {
            if let Some(ctx) = WgpuContext::global() {
                return KernelDevice::Wgpu(ctx);
            }
        }
        KernelDevice::Cpu
    }

    pub fn is_wgpu(&self) -> bool {
        matches!(self, KernelDevice::Wgpu(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, KernelDevice::Cpu)
    }
}
