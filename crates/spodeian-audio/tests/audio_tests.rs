use spodeian_audio::{
    PcmFormat, compute_acoustic_metrics, decode_wav_bytes, encode_wav_bytes,
    resample_linear_interleaved, soft_clip_tanh_in_place,
};

#[test]
fn test_wav_16_24_32f_roundtrips_and_resampling() {
    let samples: Vec<f32> = (0..128)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * i as f32 / 32.0).sin())
        .collect();

    for fmt in [PcmFormat::Pcm16, PcmFormat::Pcm24, PcmFormat::Float32] {
        let bytes = encode_wav_bytes(&samples, 1, 48_000, fmt).expect("encode wav");
        let decoded = decode_wav_bytes(&bytes).expect("decode wav");
        assert_eq!(decoded.sample_rate_hz, 48_000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples.len(), samples.len());
        let max_err = samples
            .iter()
            .zip(&decoded.samples)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 1e-3, "format {fmt:?} max_err={max_err}");
    }

    let downsampled =
        resample_linear_interleaved(&samples, 1, 48_000, 24_000).expect("resample");
    assert_eq!(downsampled.len(), 64);
}

#[test]
fn test_acoustic_metrics_and_soft_clipper() {
    let mut buf = vec![0.5, -0.5, 2.0, -3.0];
    let metrics = compute_acoustic_metrics(&buf);
    assert!(metrics.peak_dbfs > metrics.rms_dbfs);
    soft_clip_tanh_in_place(&mut buf, 1.5);
    assert!(buf.iter().all(|&s| s.is_finite() && s.abs() <= 1.0001));
}
