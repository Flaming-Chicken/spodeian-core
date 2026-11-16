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
