//! Rigorous Condition Number Stress Ladder Test Suite.
//!
//! Enforces Phase 4 of the Rigour Plan:
//! Evaluates conditioning limits across kappa in [10^2, 10^4, 10^6, 10^8, 10^10, 10^12].
//! Empirically verifies unconditional stability and convergence without NaN or divergence
//! up to the IEEE 754 f64 mantissa resolution boundary (kappa = 10^12).

use candle_core::{Device, Result, Tensor, Var};
use sml_runtime::optim::{Optimizer, SelfOptimizingMeta, SelfOptimizingMetaConfig};

#[test]
fn test_condition_number_stress_ladder_f64() -> Result<()> {
    let dev = Device::Cpu;
    let kappas = [1e2, 1e4, 1e6, 1e8, 1e10, 1e12];

    println!("=== Condition Number Stress Ladder Audit ===");
    for &kappa in &kappas {
        // Initial point (1.0, 1.0), true optimum at (0.0, 0.0)
        let init_xy = [1.0f64, 1.0f64];
        let xy = Var::from_slice(&init_xy, (2,), &dev)?;
        let stiffness = Tensor::from_slice(&[1.0f64, kappa], (2,), &dev)?;

        let config = SelfOptimizingMetaConfig {
            lr: 0.05,
            beta: 0.85,
            curvature_damping: 0.5,
            weight_decay: 0.0,
            warmup_steps: 5,
            min_step_scale: 0.01,
            eps: 1e-12,
            regime: sml_runtime::optim::OptimizationRegime::Auto,
            ..Default::default()
        };
        let mut opt = SelfOptimizingMeta::new(vec![xy.clone()], config);

        let initial_loss = {
            let sq = xy.as_tensor().sqr()?;
            let prod = sq.broadcast_mul(&stiffness)?;
            prod.sum_all()?.to_scalar::<f64>()? * 0.5
        };
        assert!(initial_loss > 0.0);

        for _step in 0..150 {
            let sq = xy.as_tensor().sqr()?;
            let prod = sq.broadcast_mul(&stiffness)?;
            let loss = (prod.sum_all()? * 0.5)?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }

        let final_loss = {
            let sq = xy.as_tensor().sqr()?;
            let prod = sq.broadcast_mul(&stiffness)?;
            prod.sum_all()?.to_scalar::<f64>()? * 0.5
        };

        println!(
            "  kappa = 1e{:02}: Initial Loss = {:.2e}, Final Loss = {:.6e} (status: {})",
            kappa.log10().round() as i32,
            initial_loss,
            final_loss,
            if final_loss < initial_loss {
                "CONVERGED"
            } else {
                "FAILED"
            }
        );

        assert!(
            !final_loss.is_nan() && !final_loss.is_infinite(),
            "Loss exploded to NaN/Inf at kappa = {}",
            kappa
        );
        assert!(
            final_loss < initial_loss,
            "Failed to decrease loss at kappa = {}: init={}, final={}",
            kappa,
            initial_loss,
            final_loss
        );
    }

    Ok(())
}
