use spodeian_forecast::{
    Garch11, KalmanFilter1D, conformal_prediction_interval, crps_gaussian, holt_linear_forecast,
    pinball_loss,
};

#[test]
fn test_holt_kalman_and_garch_forecasts() {
    let linear: Vec<f64> = (0..20).map(|t| 10.0 + 2.0 * t as f64).collect();
    let fc = holt_linear_forecast(&linear, 0.8, 0.4, 3).expect("holt linear");
    assert_eq!(fc.len(), 3);
    assert!((fc[0] - 50.0).abs() < 0.5);
    assert!((fc[2] - 54.0).abs() < 0.8);

    let mut kf = KalmanFilter1D::new(0.0, 10.0, 0.05, 0.5);
    for _ in 0..30 {
        kf.update(5.0);
    }
    assert!((kf.state - 5.0).abs() < 0.1);

    let garch = Garch11::new(1e-5, 0.1, 0.85).expect("garch");
    let resids = [0.01, -0.02, 0.03, -0.015, 0.005, -0.01];
    let (filtered, ahead) = garch.filter_and_forecast(&resids, 3).expect("garch fc");
    assert_eq!(filtered.len(), resids.len());
    assert_eq!(ahead.len(), 3);
    assert!(ahead.iter().all(|&v| v > 0.0 && v.is_finite()));
}

#[test]
fn test_conformal_pinball_and_crps() {
    let resids = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
    let (lo, hi) = conformal_prediction_interval(10.0, &resids, 0.1).expect("conformal");
    assert!(lo < 10.0 && hi > 10.0);
    assert!((hi - 10.0 - 5.0).abs() < 1e-9);

    assert!((pinball_loss(10.0, 8.0, 0.75) - 1.5).abs() < 1e-9);
    let crps_exact = crps_gaussian(0.0, 0.0, 1.0);
    let crps_off = crps_gaussian(3.0, 0.0, 1.0);
    assert!(crps_exact > 0.0 && crps_off > crps_exact);
}

/// Hand-computed with omega = 0.1, alpha = 0.2, beta = 0.5 on residuals [1, 2, 0, 1].
///
/// sigma^2_0 = mean(eps^2) = 1.5, then sigma^2_t = 0.1 + 0.2 eps_{t-1}^2 + 0.5 sigma^2_{t-1}:
/// 1.05, 1.425, 0.8125; the next (one-step-ahead) variance is 0.70625, and each
/// further step is 0.1 + 0.7 * previous.
#[test]
fn test_garch_filter_and_forecast_match_hand_computation() {
    let garch = Garch11::new(0.1, 0.2, 0.5).expect("garch");
    let (filtered, ahead) = garch
        .filter_and_forecast(&[1.0, 2.0, 0.0, 1.0], 3)
        .expect("garch fc");

    let want_filtered = [1.5, 1.05, 1.425, 0.8125];
    let want_ahead = [0.70625, 0.594375, 0.5160625];
    for (got, want) in filtered.iter().zip(want_filtered) {
        assert!((got - want).abs() < 1e-12, "filtered {got} vs {want}");
    }
    for (got, want) in ahead.iter().zip(want_ahead) {
        assert!((got - want).abs() < 1e-12, "forecast {got} vs {want}");
    }
}

#[test]
fn test_garch_forecast_reverts_to_the_long_run_variance() {
    let garch = Garch11::new(0.1, 0.2, 0.5).expect("garch");
    let (_, ahead) = garch
        .filter_and_forecast(&[1.0, 2.0, 0.0, 1.0], 200)
        .expect("garch fc");
    let long_run = 0.1 / (1.0 - 0.2 - 0.5);
    assert!((ahead[199] - long_run).abs() < 1e-9);
}

// --- Parameter estimation -----------------------------------------------------

mod estimation {
    use spodeian_forecast::{
        ForecastError, Garch11, fit_holt_linear, holt_linear_forecast, holt_one_step_sse,
    };

    /// Deterministic standard-normal noise: xorshift64* uniforms through Box-Muller.
    fn normals(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut uniform = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
            (bits as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (u1, u2) = (uniform(), uniform());
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect()
    }

    fn simulate_garch(omega: f64, alpha: f64, beta: f64, n: usize, seed: u64) -> Vec<f64> {
        let mut h = omega / (1.0 - alpha - beta);
        normals(n, seed)
            .into_iter()
            .map(|z| {
                let e = h.sqrt() * z;
                h = omega + alpha * e * e + beta * h;
                e
            })
            .collect()
    }

    /// Computed independently in Python with the same start-up convention.
    #[test]
    fn log_likelihood_matches_an_independent_calculation() {
        let model = Garch11::new(0.1, 0.2, 0.5).unwrap();
        let ll = model
            .log_likelihood(&[1.0, 2.0, 0.0, 1.0, -1.5, 0.5])
            .unwrap();
        assert!((ll - (-10.224_751_862_602)).abs() < 1e-9, "{ll}");
        assert!(matches!(
            model.log_likelihood(&[0.0; 10]),
            Err(ForecastError::InvalidParameter(_))
        ));
        assert!(model.log_likelihood(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn garch_fit_recovers_the_simulated_parameters() {
        let residuals = simulate_garch(0.05, 0.10, 0.85, 20_000, 12345);
        let fit = Garch11::fit(&residuals).unwrap();
        let m = fit.model;
        assert!(
            (m.alpha - 0.10).abs() < 0.03,
            "alpha {} (true 0.10), fit {fit:?}",
            m.alpha
        );
        assert!(
            (m.beta - 0.85).abs() < 0.04,
            "beta {} (true 0.85), fit {fit:?}",
            m.beta
        );
        assert!(
            (m.alpha + m.beta - 0.95).abs() < 0.02,
            "persistence {fit:?}"
        );
        // Variance targeting: the implied unconditional variance is the sample variance.
        let sample = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
        let unconditional = m.omega / (1.0 - m.alpha - m.beta);
        assert!((unconditional - sample).abs() < 1e-6 * sample, "{fit:?}");
        assert!(fit.log_likelihood.is_finite() && fit.iterations > 0);
    }

    /// The fit must be at least as good as every model on a grid of the same family.
    #[test]
    fn garch_fit_is_optimal_within_its_family() {
        let residuals = simulate_garch(0.2, 0.15, 0.70, 3_000, 99);
        let fit = Garch11::fit(&residuals).unwrap();
        let sample = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
        for i in 0..20 {
            for j in 0..20 {
                let alpha = 0.02 + 0.015 * i as f64;
                let beta = 0.30 + 0.034 * j as f64;
                if alpha + beta >= 0.999 {
                    continue;
                }
                let other = Garch11::new(sample * (1.0 - alpha - beta), alpha, beta).unwrap();
                let ll = other.log_likelihood(&residuals).unwrap();
                assert!(
                    fit.log_likelihood >= ll - 1e-6,
                    "grid point ({alpha}, {beta}) has ll {ll} > fit {}",
                    fit.log_likelihood
                );
            }
        }
    }

    #[test]
    fn garch_fit_rejects_unusable_input() {
        assert!(matches!(
            Garch11::fit(&[0.1; 10]),
            Err(ForecastError::InsufficientData { .. })
        ));
        assert!(matches!(
            Garch11::fit(&[0.0; 200]),
            Err(ForecastError::InvalidParameter(_))
        ));
    }

    #[test]
    fn fitted_garch_feeds_the_forecaster() {
        let residuals = simulate_garch(0.05, 0.10, 0.85, 5_000, 7);
        let fit = Garch11::fit(&residuals).unwrap();
        let (_, ahead) = fit.model.filter_and_forecast(&residuals, 5).unwrap();
        assert_eq!(ahead.len(), 5);
        assert!(ahead.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn holt_sse_matches_an_independent_calculation() {
        let series = [3.0, 5.0, 4.0, 7.0, 8.0, 6.0, 9.0];
        let sse = holt_one_step_sse(&series, 0.5, 0.3);
        assert!((sse - 23.374_969_331_602).abs() < 1e-9, "{sse}");
    }

    #[test]
    fn holt_fit_is_optimal_on_a_grid_and_usable() {
        // Noisy upward trend with a level shift.
        let noise = normals(120, 5);
        let series: Vec<f64> = noise
            .iter()
            .enumerate()
            .map(|(t, e)| 10.0 + 0.4 * t as f64 + if t > 60 { 8.0 } else { 0.0 } + 1.5 * e)
            .collect();
        let fit = fit_holt_linear(&series).unwrap();
        assert!((0.0..=1.0).contains(&fit.alpha) && (0.0..=1.0).contains(&fit.beta));
        assert!((fit.sse - holt_one_step_sse(&series, fit.alpha, fit.beta)).abs() < 1e-9);
        for i in 0..=100 {
            for j in 0..=100 {
                let sse = holt_one_step_sse(&series, i as f64 / 100.0, j as f64 / 100.0);
                assert!(
                    fit.sse <= sse + 1e-9,
                    "({}, {}) has sse {sse} < fit {}",
                    i as f64 / 100.0,
                    j as f64 / 100.0,
                    fit.sse
                );
            }
        }
        // The fitted parameters are accepted by the forecaster.
        let forecast = holt_linear_forecast(&series, fit.alpha, fit.beta, 3).unwrap();
        assert_eq!(forecast.len(), 3);
        assert!(forecast.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn holt_fit_rejects_unusable_input() {
        assert!(matches!(
            fit_holt_linear(&[1.0, 2.0, 3.0]),
            Err(ForecastError::InsufficientData { .. })
        ));
        assert!(matches!(
            fit_holt_linear(&[1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]),
            Err(ForecastError::InvalidParameter(_))
        ));
    }
}
