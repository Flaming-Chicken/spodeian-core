//! Phase P6: Counting Allocator Verification Gate.
//! Asserts that native coordinate updates execute with strictly ZERO heap allocations during steady state.

use sml_runtime::optim_kernels::{update_coordinates_f32, KernelUpdateParams};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct TrackingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACKING_ACTIVE: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING_ACTIVE.load(Ordering::SeqCst) {
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: TrackingAllocator = TrackingAllocator;

#[test]
fn test_zero_allocation_during_coordinate_update_steady_state() {
    let n = 4096;
    let mut y = vec![1.5f32; n];
    let mut z = vec![1.5f32; n];
    let mut m = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    let mut prev_g = vec![0.0f32; n];
    let mut flip_ema = vec![0.0f32; n];
    let grads = vec![0.25f32; n];

    let params = KernelUpdateParams {
        step: 0,
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        beta_polyak: 0.85,
        gamma_flip: 0.9,
        eps: 1e-8,
        weight_decay: 0.0,
        cusp_power: 1.0,
        lambda_hill: 0.0,
        flip_damping_factor: 0.5,
        mean_g_sq: 0.0,
        is_multi_layer: false,
    };

    // Warm-up step
    update_coordinates_f32(
        params,
        &mut y,
        &mut z,
        &mut m,
        &mut v,
        &mut prev_g,
        &mut flip_ema,
        &grads,
    );

    // Activate tracking allocator
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKING_ACTIVE.store(true, Ordering::SeqCst);

    // Run 50 steady-state optimization steps
    for s in 1..=50 {
        let mut p = params;
        p.step = s;
        update_coordinates_f32(
            p,
            &mut y,
            &mut z,
            &mut m,
            &mut v,
            &mut prev_g,
            &mut flip_ema,
            &grads,
        );
    }

    TRACKING_ACTIVE.store(false, Ordering::SeqCst);
    let total_allocs = ALLOC_COUNT.load(Ordering::SeqCst);

    println!(
        "Steady-state coordinate updates (50 steps x 4096 params): {} heap allocations",
        total_allocs
    );

    assert_eq!(
        total_allocs, 0,
        "Gate Failure: Coordinate update loop must perform strictly ZERO heap allocations!"
    );
}
