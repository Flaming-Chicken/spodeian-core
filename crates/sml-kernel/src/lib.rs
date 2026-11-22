//! Hardware compute kernel runtime (`sml-kernel`): WebGPU (WGSL) pipeline cache & CPU/WebGPU device abstraction.

pub mod context;
#[cfg(feature = "wgpu")]
pub mod wgpu_util;

pub use context::{KernelBackendSummary, KernelDevice, WgpuContext};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_device_cpu_and_auto() {
        let cpu = KernelDevice::Cpu;
        assert!(cpu.is_cpu());
        assert!(!cpu.is_wgpu());
        let summary = cpu.summary();
        assert_eq!(summary.backend, "cpu-fallback");
        assert!(!summary.is_wgpu_active);
        assert!(!summary.supports_wgsl_compute);
        assert_eq!(summary.max_workgroup_size_x, 0);

        let auto_dev = KernelDevice::auto();
        assert!(auto_dev.is_cpu() || auto_dev.is_wgpu());
    }
}
