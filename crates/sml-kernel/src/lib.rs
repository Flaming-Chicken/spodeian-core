//! Hardware compute kernel runtime (`sml-kernel`): WebGPU (WGSL) pipeline cache & CPU/WebGPU device abstraction.

pub mod context;
#[cfg(feature = "wgpu")]
pub mod wgpu_util;

pub use context::{KernelDevice, WgpuContext};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_device_cpu_and_auto() {
        let cpu = KernelDevice::Cpu;
        assert!(cpu.is_cpu());
        assert!(!cpu.is_wgpu());
        let auto_dev = KernelDevice::auto();
        assert!(auto_dev.is_cpu() || auto_dev.is_wgpu());
    }
}
