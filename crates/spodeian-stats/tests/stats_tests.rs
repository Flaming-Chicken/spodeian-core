use spodeian_stats::{
    ObjectiveDirection, adf_statistic, chi2_sf, engle_granger_cointegration,
    exact_non_dominated_indices, huber_irls_regression, hypervolume_2d, is_strict_pareto_frontier,
    jarque_bera_stat, newey_west_hac, normal_cdf, normal_ppf, ols, ridge_regression,
    select_utopia_knee_index, student_t_cdf,
};

#[test]
fn test_distributions_and_quantiles() {
    assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
    assert!((normal_cdf(1.96) - 0.975_002).abs() < 1e-3);

    let z975 = normal_ppf(0.975).expect("normal_ppf(0.975)");
    assert!((z975 - 1.959_964).abs() < 1e-4);
    assert!((normal_ppf(0.025).expect("lower tail") + 1.959_964).abs() < 1e-4);

    let t_p = student_t_cdf(0.0, 10.0).expect("t(0)");
    assert!((t_p - 0.5).abs() < 1e-8);
    let t_hi = student_t_cdf(2.228_139, 10.0).expect("t(0.975, df=10)");
    assert!((t_hi - 0.975).abs() < 1e-3);

    let sf = chi2_sf(3.841_459, 1.0).expect("chi2_sf(3.841, 1)");
    assert!((sf - 0.05).abs() < 1e-3);
}

#[test]
fn test_ols_ridge_hac_and_huber_outlier_resistance() {
    let mut x_rows = Vec::new();
    let mut y_clean = Vec::new();
    let mut y_outlier = Vec::new();
    for i in 0..40 {
        let x = i as f64 * 0.25;
        let y = 1.0 + 2.0 * x;
        x_rows.push(vec![1.0, x]);
        y_clean.push(y);
        y_outlier.push(if i == 20 { y + 100.0 } else { y });
    }

    let ols_res = ols(&y_clean, &x_rows).expect("ols");
    assert!((ols_res.beta[0] - 1.0).abs() < 1e-9);
    assert!((ols_res.beta[1] - 2.0).abs() < 1e-9);

    let ridge_res = ridge_regression(&y_clean, &x_rows, 10.0).expect("ridge");
    let ols_norm2: f64 = ols_res.beta.iter().map(|b| b * b).sum();
    let ridge_norm2: f64 = ridge_res.beta.iter().map(|b| b * b).sum();
    assert!(
        ridge_norm2 < ols_norm2,
        "expected ridge L2 norm ({ridge_norm2}) < OLS L2 norm ({ols_norm2})"
    );

    let hac_res = newey_west_hac(&y_clean, &x_rows, 2).expect("hac");
    assert_eq!(hac_res.beta.len(), 2);

    let ols_corrupted = ols(&y_outlier, &x_rows).expect("ols corrupted");
    let huber_res = huber_irls_regression(&y_outlier, &x_rows, 1.345, 25).expect("huber");
    assert!(
        (huber_res.beta[1] - 2.0).abs() < (ols_corrupted.beta[1] - 2.0).abs(),
        "Huber should resist outlier better than OLS: huber={}, ols={}",
        huber_res.beta[1],
        ols_corrupted.beta[1]
    );
}

#[test]
fn test_jarque_bera_and_engle_granger_cointegration() {
    let symmetric: Vec<f64> = (0..40)
        .map(|i| {
            let z = (i as f64 - 19.5) / 10.0;
            if i % 2 == 0 { z * 0.1 } else { -z * 0.1 }
        })
        .collect();
    let skewed: Vec<f64> = (0..40).map(|i| if i < 38 { 0.01 } else { 5.0 }).collect();
    let jb_sym = jarque_bera_stat(&symmetric).expect("JB symmetric");
    let jb_skew = jarque_bera_stat(&skewed).expect("JB skewed");
    assert!(jb_skew > jb_sym * 10.0);

    // Cointegrated pair: y_t = 2 * x_t + stationary_ar1_noise_t
    let mut x = Vec::with_capacity(80);
    let mut y = Vec::with_capacity(80);
    let mut rw = 0.0_f64;
    let mut noise = 0.0_f64;
    for t in 0..80 {
        rw += if t % 3 == 0 { 0.5 } else { -0.2 };
        let eps = ((t % 5) as f64 - 2.0) * 0.04 + ((t % 3) as f64 - 1.0) * 0.02;
        noise = 0.25 * noise + eps;
        x.push(rw);
        y.push(2.0 * rw + noise);
    }
    let (slope, resid_adf) = engle_granger_cointegration(&y, &x, 1).expect("cointegration");
    assert!((slope - 2.0).abs() < 0.05);
    assert!(
        resid_adf < -3.0,
        "expected stationary residual ADF < -3, got {resid_adf}"
    );
    assert!(adf_statistic(&x, 1).expect("rw adf") > resid_adf);
}

#[test]
fn test_pareto_exact_dominance_knee_and_hypervolume() {
    // Objectives: [minimize cost, maximize accuracy]
    let dirs = [ObjectiveDirection::Minimize, ObjectiveDirection::Maximize];
    let pts = vec![
        vec![1.0, 0.70], // non-dominated
        vec![2.0, 0.85], // knee
        vec![2.5, 0.80], // dominated by [2.0, 0.85]
        vec![4.0, 0.92], // non-dominated
    ];
    let idx = exact_non_dominated_indices(&pts, &dirs);
    assert_eq!(idx, vec![0, 1, 3]);

    let frontier: Vec<Vec<f64>> = idx.iter().map(|&i| pts[i].clone()).collect();
    assert!(is_strict_pareto_frontier(&frontier, &dirs));
    assert_eq!(select_utopia_knee_index(&frontier, &dirs), Some(1));

    let hv = hypervolume_2d(&[(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)], (5.0, 5.0));
    assert!((hv - 11.0).abs() < 1e-9);
}

// --- ADF critical values and p-values -----------------------------------------

mod adf {
    use spodeian_stats::{
        AdfRegression, StatsError, adf_critical_value, adf_p_value, adf_statistic, adf_test,
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

    fn random_walk(n: usize, seed: u64) -> Vec<f64> {
        normals(n, seed)
            .into_iter()
            .scan(0.0, |acc, e| {
                *acc += e;
                Some(*acc)
            })
            .collect()
    }

    fn ar1(n: usize, rho: f64, seed: u64) -> Vec<f64> {
        normals(n, seed)
            .into_iter()
            .scan(0.0, |y, e| {
                *y = rho * *y + e;
                Some(*y)
            })
            .collect()
    }

    const ALL: [AdfRegression; 3] = [
        AdfRegression::NoConstant,
        AdfRegression::Constant,
        AdfRegression::ConstantTrend,
    ];

    /// Values computed independently from the published coefficients with Python.
    #[test]
    fn critical_values_match_the_response_surface() {
        let cases = [
            (0.05, AdfRegression::Constant, 100, -2.890_906_440_0),
            (0.01, AdfRegression::ConstantTrend, 250, -3.995_445_833_9),
            (0.10, AdfRegression::NoConstant, 50, -1.612_390_688_0),
        ];
        for (level, regression, n_obs, expected) in cases {
            let got = adf_critical_value(level, regression, n_obs).unwrap();
            assert!(
                (got - expected).abs() < 1e-9,
                "{regression:?} {level} T={n_obs}: {got} vs {expected}"
            );
        }
        // Critical values tighten monotonically with the level and approach the asymptote.
        for regression in ALL {
            let one = adf_critical_value(0.01, regression, 500).unwrap();
            let five = adf_critical_value(0.05, regression, 500).unwrap();
            let ten = adf_critical_value(0.10, regression, 500).unwrap();
            assert!(
                one < five && five < ten,
                "{regression:?}: {one} {five} {ten}"
            );
        }
    }

    /// The p-value polynomial must return the nominal level at the (asymptotic) critical
    /// value; this ties the two sets of published coefficients to each other.
    #[test]
    fn p_value_equals_the_level_at_the_critical_value() {
        for regression in ALL {
            for level in [0.01, 0.05, 0.10] {
                let tau = adf_critical_value(level, regression, 1_000_000_000).unwrap();
                let p = adf_p_value(tau, regression);
                assert!(
                    (p - level).abs() < 6e-4,
                    "{regression:?}: p({tau}) = {p}, expected {level}"
                );
            }
        }
    }

    #[test]
    fn p_value_branches_and_bounds() {
        let cases = [
            // large-p branch, then small-p branch; exact normal CDF of the polynomial
            (0.0, AdfRegression::Constant, 0.958_532_086_1),
            (-1.0, AdfRegression::Constant, 0.753_264_301_2),
            (-3.0, AdfRegression::ConstantTrend, 0.132_080_984_8),
            (-0.5, AdfRegression::NoConstant, 0.496_124_037_5),
        ];
        for (tau, regression, expected) in cases {
            let got = adf_p_value(tau, regression);
            // The normal CDF used is accurate to about 1.5e-7.
            assert!(
                (got - expected).abs() < 1e-6,
                "{regression:?} p({tau}) = {got}, expected {expected}"
            );
        }
        assert_eq!(adf_p_value(10.0, AdfRegression::Constant), 1.0);
        assert_eq!(adf_p_value(-30.0, AdfRegression::Constant), 0.0);
        // Monotone: a more negative statistic is stronger evidence against a unit root.
        for regression in ALL {
            let mut previous = 0.0;
            for k in 0..60 {
                let p = adf_p_value(-6.0 + 0.1 * k as f64, regression);
                assert!(p >= previous - 1e-9, "{regression:?} not monotone at {k}");
                previous = p;
            }
        }
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        assert!(matches!(
            adf_critical_value(0.025, AdfRegression::Constant, 100),
            Err(StatsError::InvalidParameter(_))
        ));
        assert!(matches!(
            adf_critical_value(0.05, AdfRegression::Constant, 0),
            Err(StatsError::InsufficientData { .. })
        ));
        assert!(adf_test(&[1.0, 2.0, 3.0], 1, AdfRegression::Constant).is_err());
    }

    /// With no lags and a constant the regression is simple: dy on [1, y_{t-1}].
    #[test]
    fn statistic_matches_a_direct_simple_regression() {
        let series = random_walk(300, 7);
        let dy: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
        let x = &series[..series.len() - 1];
        let n = dy.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, dy.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(&dy).map(|(a, b)| (a - mx) * (b - my)).sum();
        let slope = sxy / sxx;
        let intercept = my - slope * mx;
        let sse: f64 = x
            .iter()
            .zip(&dy)
            .map(|(a, b)| (b - intercept - slope * a).powi(2))
            .sum();
        let t_expected = slope / (sse / (n - 2.0) / sxx).sqrt();

        let result = adf_test(&series, 0, AdfRegression::Constant).unwrap();
        assert!(
            (result.statistic - t_expected).abs() < 1e-9,
            "{} vs {t_expected}",
            result.statistic
        );
        assert_eq!(result.n_obs, 299);
        assert!((adf_statistic(&series, 0).unwrap() - result.statistic).abs() < 1e-12);
    }

    #[test]
    fn statistic_ignores_the_deterministic_terms_it_includes() {
        let base = random_walk(400, 11);
        let reference_c = adf_test(&base, 2, AdfRegression::Constant)
            .unwrap()
            .statistic;
        let reference_ct = adf_test(&base, 2, AdfRegression::ConstantTrend)
            .unwrap()
            .statistic;

        // A shift in level does not matter when the regression has a constant.
        let shifted: Vec<f64> = base.iter().map(|v| v + 40.0).collect();
        let shifted_stat = adf_test(&shifted, 2, AdfRegression::Constant)
            .unwrap()
            .statistic;
        assert!((shifted_stat - reference_c).abs() < 1e-7);

        // A linear trend does not matter when the regression has a trend term.
        let trended: Vec<f64> = base
            .iter()
            .enumerate()
            .map(|(t, v)| v + 0.3 * t as f64)
            .collect();
        let trended_stat = adf_test(&trended, 2, AdfRegression::ConstantTrend)
            .unwrap()
            .statistic;
        assert!((trended_stat - reference_ct).abs() < 1e-6);

        // Scale does not matter.
        let scaled: Vec<f64> = base.iter().map(|v| 5.0 * v).collect();
        let scaled_stat = adf_test(&scaled, 2, AdfRegression::Constant)
            .unwrap()
            .statistic;
        assert!((scaled_stat - reference_c).abs() < 1e-7);
    }

    #[test]
    fn stationary_series_reject_and_random_walks_do_not() {
        let stationary = adf_test(&ar1(1000, 0.5, 3), 1, AdfRegression::Constant).unwrap();
        assert!(stationary.rejects_unit_root_at_5pct(), "{stationary:?}");
        assert!(stationary.p_value < 0.01, "{stationary:?}");
        assert!(stationary.critical_1pct < stationary.critical_5pct);

        let walk = adf_test(&random_walk(1000, 3), 1, AdfRegression::Constant).unwrap();
        assert!(!walk.rejects_unit_root_at_5pct(), "{walk:?}");
        assert!(walk.p_value > 0.10, "{walk:?}");
        assert_eq!(walk.n_obs, 998);
    }
}
