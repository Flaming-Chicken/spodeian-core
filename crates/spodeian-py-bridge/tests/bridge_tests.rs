use spodeian_py_bridge::{
    apply_gaussian_jitter, apply_temporal_mask, calculate_ema, calculate_log_returns,
    calculate_rolling_volatility, calculate_vwap, rolling_zscore_normalize, time_warp_resample,
};

#[test]
fn test_temporal_masking_reproducibility() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let masked1 = apply_temporal_mask(&input, 0.4, -999.0, 42);
    let masked2 = apply_temporal_mask(&input, 0.4, -999.0, 42);

    assert_eq!(
        masked1, masked2,
        "Masking must be deterministic for identical seeds"
    );
    assert!(
        masked1.contains(&-999.0),
        "Masking should replace elements with mask_value"
    );
}

#[test]
fn test_gaussian_jitter_properties() {
    let input = vec![10.0; 100];
    let jittered = apply_gaussian_jitter(&input, 0.1, 1234);

    assert_eq!(jittered.len(), 100);
    // Values should fluctuate closely around 10.0
    let mean = jittered.iter().sum::<f64>() / 100.0;
    assert!(
        (mean - 10.0).abs() < 0.05,
        "Mean of jittered constant should remain close to original"
    );
}

#[test]
fn test_rolling_zscore() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let z = rolling_zscore_normalize(&input, 3);

    assert_eq!(z.len(), 5);
    // Last element: slice [3.0, 4.0, 5.0], mean=4.0, std=sqrt(2/3)=0.81649, z = (5.0-4.0)/0.81649 = 1.2247
    let last = *z.last().unwrap();
    assert!(last > 1.0 && last < 1.3);
}

#[test]
fn test_time_warp_resample() {
    let input = vec![0.0, 10.0];
    let resampled = time_warp_resample(&input, 5);

    assert_eq!(resampled.len(), 5);
    assert_eq!(resampled[0], 0.0);
    assert_eq!(resampled[4], 10.0);
    assert_eq!(resampled[2], 5.0); // Exact midpoint
}

#[test]
fn test_financial_factor_calculations() {
    let prices = vec![100.0, 105.0, 102.0, 110.0];
    let returns = calculate_log_returns(&prices);
    assert_eq!(returns.len(), 4);
    assert_eq!(returns[0], 0.0);
    assert!((returns[1] - (105.0_f64 / 100.0).ln()).abs() < 1e-6);

    let vol = calculate_rolling_volatility(&returns, 3);
    assert_eq!(vol.len(), 4);

    let ema = calculate_ema(&prices, 2);
    assert_eq!(ema.len(), 4);
    assert_eq!(ema[0], 100.0);

    let volumes = vec![10.0, 20.0, 15.0, 30.0];
    let vwap = calculate_vwap(&prices, &volumes);
    assert_eq!(vwap.len(), 4);
    assert_eq!(vwap[0], 100.0);
    // VWAP step 1: (100*10 + 105*20) / 30 = 3100 / 30 = 103.333
    assert!((vwap[1] - 103.333333).abs() < 1e-4);
}
