//! Device and context management for CPU and WebGPU backends.

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

/// Cached singleton WebGPU context across threads/tests to prevent graphics driver lock contention.
#[cfg(not(target_arch = "wasm32"))]
static GLOBAL_WGPU_CONTEXT: OnceLock<Option<WgpuContext>> = OnceLock::new();

/// Serializable capability summary describing the active [`KernelDevice`] execution backend.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KernelBackendSummary {
    /// Human-readable backend identifier (`"wgpu"` or `"cpu-fallback"`).
    pub backend: String,
    /// Whether a hardware WebGPU adapter and device are active.
    pub is_wgpu_active: bool,
    /// Whether WGSL compute shaders can be dispatched on this device.
    pub supports_wgsl_compute: bool,
    /// Reported maximum compute workgroup X dimension (`0` when running on CPU fallback).
    pub max_workgroup_size_x: u32,
}

/// WebGPU device and queue context.
#[derive(Debug, Clone)]
pub struct WgpuContext {
    /// Active WebGPU logical device handle.
    #[cfg(feature = "wgpu")]
    pub device: Arc<wgpu::Device>,
    /// Active WebGPU command queue handle.
    #[cfg(feature = "wgpu")]
    pub queue: Arc<wgpu::Queue>,
}

impl WgpuContext {
    /// Returns the global singleton WebGPU context, initializing it once if available.
    ///
    /// On wasm32 this is always `None` (CPU fallback): device setup blocks on
    /// `pollster`, which can't drive the browser's WebGPU futures, and wgpu
    /// handles there are neither `Send` nor `Sync`. Use [`WgpuContext::new_async`]
    /// from browser/Tauri async tasks instead.
    pub fn global() -> Option<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        return GLOBAL_WGPU_CONTEXT.get_or_init(Self::new).clone();
        #[cfg(target_arch = "wasm32")]
        None
    }

    /// Asynchronously requests a high-performance WebGPU adapter and logical device across both
    /// native platforms and `wasm32` browsers.
    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
    pub async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ai-kernel-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Asynchronous WebGPU initializer stub when `feature = "wgpu"` is disabled.
    #[cfg(not(feature = "wgpu"))]
    pub async fn new_async() -> Option<Self> {
        None
    }

    /// Blocking convenience constructor delegating to [`WgpuContext::new_async`] on native targets.
    ///
    /// Returns `None` on `wasm32` (browsers must await [`WgpuContext::new_async`] instead).
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Blocking constructor stub returning `None` when WebGPU is disabled or on `wasm32`.
    #[cfg(any(not(feature = "wgpu"), target_arch = "wasm32"))]
    pub fn new() -> Option<Self> {
        None
    }
}

/// Execution device abstraction: CPU or WebGPU with automatic fallback.
#[derive(Debug, Clone)]
pub enum KernelDevice {
    /// Portable CPU / SIMD execution path.
    Cpu,
    /// Hardware-accelerated WebGPU compute path.
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

    /// Returns `true` when backed by an active [`WgpuContext`].
    pub fn is_wgpu(&self) -> bool {
        matches!(self, KernelDevice::Wgpu(_))
    }

    /// Returns `true` when executing via the CPU fallback path.
    pub fn is_cpu(&self) -> bool {
        matches!(self, KernelDevice::Cpu)
    }

    /// Returns a serializable capability summary describing this compute device.
    pub fn summary(&self) -> KernelBackendSummary {
        match self {
            Self::Cpu => KernelBackendSummary {
                backend: "cpu-fallback".to_string(),
                is_wgpu_active: false,
                supports_wgsl_compute: false,
                max_workgroup_size_x: 0,
            },
            #[cfg(feature = "wgpu")]
            Self::Wgpu(ctx) => KernelBackendSummary {
                backend: "wgpu".to_string(),
                is_wgpu_active: true,
                supports_wgsl_compute: true,
                max_workgroup_size_x: ctx.device.limits().max_compute_workgroup_size_x,
            },
            #[cfg(not(feature = "wgpu"))]
            Self::Wgpu(_) => KernelBackendSummary {
                backend: "wgpu-disabled".to_string(),
                is_wgpu_active: false,
                supports_wgsl_compute: false,
                max_workgroup_size_x: 0,
            },
        }
    }
}
