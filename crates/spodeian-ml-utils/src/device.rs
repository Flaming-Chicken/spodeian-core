//! Universal Compute Device, Capability Discovery, and Hardware Execution Router.
//!
//! **Stability: Experimental** — API subject to change
//!
//! Standardizes hardware execution and device selection across the entire Spodeian ecosystem:
//! - [`ComputeDevice`]: Universal device taxonomy (CPU, CUDA GPU, Apple Silicon Metal, WASM SIMD, WebGPU).
//! - [`ComputeCapabilities`]: Hardware capability matrix (half precision, vector width, thread count).
//! - [`ComputeRouter`]: Automatic hardware discovery, capability inspection, and Candle/PyTorch device routing.

use serde::{Deserialize, Serialize};

/// Universal hardware compute devices supported across Spodeian engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ComputeDevice {
    /// Host CPU (fallback architecture supported everywhere).
    #[default]
    Cpu,
    /// NVIDIA CUDA GPU with specified device ordinal index.
    Cuda(usize),
    /// Apple Silicon GPU via Metal Performance Shaders (MPS).
    Metal,
    /// WebAssembly 128-bit SIMD in modern browsers and WASI runtimes.
    WasmSimd,
    /// Browser WebGPU compute shader pipeline (WGSL).
    WebGpu,
}

impl ComputeDevice {
    /// Human-readable label for user interfaces and logs.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Cpu => "Host CPU (Single/Multi-threaded)",
            Self::Cuda(_) => "NVIDIA CUDA GPU Accelerator",
            Self::Metal => "Apple Silicon GPU (Metal)",
            Self::WasmSimd => "WebAssembly 128-bit SIMD",
            Self::WebGpu => "WebGPU Compute Shader Pipeline",
        }
    }

    /// Short standard identifier string (compatible with PyTorch and Candle).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda(_) => "cuda",
            Self::Metal => "metal",
            Self::WasmSimd => "wasm-simd",
            Self::WebGpu => "webgpu",
        }
    }

    /// Whether this device is a GPU or hardware accelerator.
    #[inline(always)]
    pub fn is_accelerated(&self) -> bool {
        !matches!(self, Self::Cpu | Self::WasmSimd)
    }

    /// Converts into a [`candle_core::Device`].
    pub fn to_candle_device(&self) -> Result<candle_core::Device, String> {
        match self {
            Self::Cpu | Self::WasmSimd | Self::WebGpu => Ok(candle_core::Device::Cpu),
            Self::Cuda(ord) => {
                #[cfg(feature = "cuda")]
                {
                    candle_core::Device::new_cuda(*ord).map_err(|e| e.to_string())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = ord;
                    Err("CUDA feature not compiled into this binary".to_string())
                }
            }
            Self::Metal => {
                #[cfg(feature = "metal")]
                {
                    candle_core::Device::new_metal(0).map_err(|e| e.to_string())
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal feature not compiled into this binary".to_string())
                }
            }
        }
    }
}

/// Hardware performance capabilities and feature flags for a compute device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeCapabilities {
    pub device: ComputeDevice,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub simd_vector_bits: usize,
    pub max_threads: usize,
    pub description: String,
}

/// Central hardware discovery and compute routing engine.
#[derive(Debug, Clone, Default)]
pub struct ComputeRouter;

impl ComputeRouter {
    /// Discovers and returns the highest-performance available hardware device.
    ///
    /// Fallback hierarchy: CUDA -> Metal -> WasmSimd -> CPU.
    pub fn discover_best_device() -> ComputeDevice {
        #[cfg(target_arch = "wasm32")]
        {
            #[cfg(target_feature = "simd128")]
            {
                ComputeDevice::WasmSimd
            }
            #[cfg(not(target_feature = "simd128"))]
            {
                ComputeDevice::Cpu
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            #[cfg(feature = "cuda")]
            {
                if candle_core::Device::new_cuda(0).is_ok() {
                    return ComputeDevice::Cuda(0);
                }
            }

            #[cfg(feature = "metal")]
            {
                if candle_core::Device::new_metal(0).is_ok() {
                    return ComputeDevice::Metal;
                }
            }

            ComputeDevice::Cpu
        }
    }

    /// Queries the hardware capabilities of the target device.
    pub fn query_capabilities(device: ComputeDevice) -> ComputeCapabilities {
        match device {
            ComputeDevice::Cuda(ord) => ComputeCapabilities {
                device,
                supports_fp16: true,
                supports_bf16: true,
                simd_vector_bits: 256,
                max_threads: 1024,
                description: format!("NVIDIA CUDA GPU Ordinal #{ord}"),
            },
            ComputeDevice::Metal => ComputeCapabilities {
                device,
                supports_fp16: true,
                supports_bf16: true,
                simd_vector_bits: 128,
                max_threads: 512,
                description: "Apple Silicon Metal Acceleration Pipeline".to_string(),
            },
            ComputeDevice::WasmSimd => ComputeCapabilities {
                device,
                supports_fp16: false,
                supports_bf16: false,
                simd_vector_bits: 128,
                max_threads: 1,
                description: "WebAssembly 128-bit SIMD Web Engine".to_string(),
            },
            ComputeDevice::WebGpu => ComputeCapabilities {
                device,
                supports_fp16: true,
                supports_bf16: false,
                simd_vector_bits: 128,
                max_threads: 256,
                description: "WebGPU WGSL Compute Pipeline".to_string(),
            },
            ComputeDevice::Cpu => {
                let threads = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1);
                ComputeCapabilities {
                    device,
                    supports_fp16: false,
                    supports_bf16: false,
                    simd_vector_bits: 256,
                    max_threads: threads,
                    description: format!("Host CPU ({threads} hardware threads)"),
                }
            }
        }
    }

    /// Parses a standard device descriptor string (e.g. "cpu", "cuda:0", "metal", "wasm").
    pub fn parse_device(s: &str) -> Result<ComputeDevice, String> {
        let clean = s.trim().to_lowercase();
        if clean == "cpu" {
            Ok(ComputeDevice::Cpu)
        } else if clean == "metal" || clean == "mps" {
            Ok(ComputeDevice::Metal)
        } else if clean == "wasm" || clean == "wasm-simd" {
            Ok(ComputeDevice::WasmSimd)
        } else if clean == "webgpu" {
            Ok(ComputeDevice::WebGpu)
        } else if clean == "cuda" {
            Ok(ComputeDevice::Cuda(0))
        } else if let Some(stripped) = clean.strip_prefix("cuda:") {
            let ord = stripped
                .parse::<usize>()
                .map_err(|_| format!("Invalid CUDA ordinal index: '{stripped}'"))?;
            Ok(ComputeDevice::Cuda(ord))
        } else {
            Err(format!("Unknown compute device string: '{s}'"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_string_parsing() {
        assert_eq!(ComputeRouter::parse_device("cpu").unwrap(), ComputeDevice::Cpu);
        assert_eq!(ComputeRouter::parse_device("cuda").unwrap(), ComputeDevice::Cuda(0));
        assert_eq!(ComputeRouter::parse_device("cuda:2").unwrap(), ComputeDevice::Cuda(2));
        assert_eq!(ComputeRouter::parse_device("metal").unwrap(), ComputeDevice::Metal);
        assert_eq!(ComputeRouter::parse_device("mps").unwrap(), ComputeDevice::Metal);
        assert_eq!(ComputeRouter::parse_device("webgpu").unwrap(), ComputeDevice::WebGpu);
        assert!(ComputeRouter::parse_device("quantum").is_err());
    }

    #[test]
    fn test_device_discovery_and_capabilities() {
        let best = ComputeRouter::discover_best_device();
        let caps = ComputeRouter::query_capabilities(best);
        assert!(caps.max_threads >= 1);
        assert!(!caps.description.is_empty());
        let candle_dev = best.to_candle_device();
        assert!(candle_dev.is_ok());
    }
}
