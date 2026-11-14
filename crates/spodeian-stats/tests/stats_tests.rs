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
