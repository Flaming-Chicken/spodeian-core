use criterion::{Criterion, black_box, criterion_group, criterion_main};
use sml_runtime::optim::{SelfOptimizingMeta, SelfOptimizingMetaConfig};
use sml_runtime::solvers::SelfOptimizingMetaIntegrator;

fn bench_meta_optimizer_step(c: &mut Criterion) {
    let dim = 1024;
    let grads = vec![0.05f32; dim];
    c.bench_function("SelfOptimizingMeta_step_1024f32", |b| {
        let mut opt = SelfOptimizingMeta::new(Vec::new(), SelfOptimizingMetaConfig::default());
        let mut theta = vec![1.0f32; dim];
        b.iter(|| {
            let _ = opt.step_slice_in_place_f32(0, black_box(&grads), black_box(&mut theta));
        });
    });
}

fn bench_meta_integrator_harmonic(c: &mut Criterion) {
    let solver = SelfOptimizingMetaIntegrator::new(1e-4);
    c.bench_function("SelfOptimizingMetaIntegrator_harmonic_0_to_2", |b| {
        b.iter(|| {
            let (xf, _diag, _nfe) =
                solver.solve_interval(black_box(&[1.0f32, 0.0f32]), 0.0, 2.0, |s, _t| {
                    vec![s[1], -s[0]]
                });
            black_box(xf);
        });
    });
}

criterion_group!(
    benches,
    bench_meta_optimizer_step,
    bench_meta_integrator_harmonic
);
criterion_main!(benches);
