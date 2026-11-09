//! Procedural domain data generators and canonical tabular schema traits.
//!
//! **Stability: Experimental** — API subject to change
//!
//! Provides deterministic, zero-dependency synthetic data generators for:
//! - Financial market time-series, order book ticks, and OHLCV bars.
//! - Audio acoustic telemetry snapshots and sinusoidal chirp test signals.
//! - Reusable [`TabularSchema`] trait unifying struct-to-Arrow conversions.

use crate::parquet_io::{ParquetError, TabularDataset};
use crate::schema::{AcousticTelemetrySnapshot, FinancialObservation, PlayerProgressEvent};

/// Generalized bidirectional schema mapping between typed domain records and columnar [`TabularDataset`].
pub trait TabularSchema: Sized {
    /// Ordered list of column identifiers.
    fn column_names() -> &'static [&'static str];

    /// Serializes a slice of domain records into a columnar [`TabularDataset`].
    fn to_tabular_dataset(records: &[Self]) -> Result<TabularDataset, ParquetError>;

    /// Reconstructs a list of typed domain records from a [`TabularDataset`].
    fn from_tabular_dataset(dataset: &TabularDataset) -> Result<Vec<Self>, ParquetError>;
}

impl TabularSchema for FinancialObservation {
    fn column_names() -> &'static [&'static str] {
        &[
            "timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "spread",
        ]
    }

    fn to_tabular_dataset(records: &[Self]) -> Result<TabularDataset, ParquetError> {
        FinancialObservation::to_tabular_dataset(records)
    }

    fn from_tabular_dataset(dataset: &TabularDataset) -> Result<Vec<Self>, ParquetError> {
        FinancialObservation::from_tabular_dataset(dataset)
    }
}

impl TabularSchema for AcousticTelemetrySnapshot {
    fn column_names() -> &'static [&'static str] {
        &[
            "timestamp",
            "frame_idx",
            "rms_db",
            "peak_db",
            "spectral_centroid",
            "zero_crossing_rate",
            "active_voices",
            "buffer_underruns",
        ]
    }

    fn to_tabular_dataset(records: &[Self]) -> Result<TabularDataset, ParquetError> {
        AcousticTelemetrySnapshot::to_tabular_dataset(records)
    }

    fn from_tabular_dataset(dataset: &TabularDataset) -> Result<Vec<Self>, ParquetError> {
        AcousticTelemetrySnapshot::from_tabular_dataset(dataset)
    }
}

impl TabularSchema for PlayerProgressEvent {
    fn column_names() -> &'static [&'static str] {
        &[
            "timestamp",
            "player_id",
            "level",
            "score",
            "latency_ms",
            "quest_id",
        ]
    }

    fn to_tabular_dataset(records: &[Self]) -> Result<TabularDataset, ParquetError> {
        PlayerProgressEvent::to_tabular_dataset(records)
    }

    fn from_tabular_dataset(dataset: &TabularDataset) -> Result<Vec<Self>, ParquetError> {
        PlayerProgressEvent::from_tabular_dataset(dataset)
    }
}

/// Simple 64-bit Xorshift PRNG for fast, deterministic, reproducible synthetic simulation.
#[derive(Debug, Clone)]
struct SimplePrng {
    state: u64,
}

impl SimplePrng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Approximate standard normal distribution $N(0, 1)$ via Box-Muller transform.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Procedural financial market data generator simulating Geometric Brownian Motion and micro-structure spreads.
#[derive(Debug, Clone)]
pub struct SyntheticMarketGenerator {
    seed: u64,
}

impl SyntheticMarketGenerator {
    /// Creates a new market generator with the specified deterministic seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generates a series of realistic [`FinancialObservation`] price bars.
    pub fn generate_ticks(
        &self,
        symbol: &str,
        count: usize,
        start_price: f64,
        annualized_vol: f64,
    ) -> Vec<FinancialObservation> {
        let mut prng = SimplePrng::new(self.seed);
        let mut observations = Vec::with_capacity(count);

        let dt = 1.0f64 / 252.0f64; // Daily step
        let dt_sqrt = dt.sqrt();
        let mu = 0.08f64; // 8% drift
        let mut current_price = start_price;
        let start_time = 1700000000i64;

        for i in 0..count {
            let z = prng.next_gaussian();
            let return_step = (mu - 0.5 * annualized_vol * annualized_vol) * dt
                + annualized_vol * dt_sqrt * z;
            let close_price = (current_price * return_step.exp()).max(0.01);

            let high_bump = prng.next_f64().abs() * 0.01 * close_price;
            let low_dip = prng.next_f64().abs() * 0.01 * close_price;

            let open = current_price;
            let high = open.max(close_price) + high_bump;
            let low = (open.min(close_price) - low_dip).max(0.01);
            let volume = (1000.0 + prng.next_f64() * 50000.0).round();
            let vwap = (open + high + low + close_price) / 4.0;
            let spread = (close_price * 0.0005 * (1.0 + prng.next_f64())).max(0.01);

            observations.push(FinancialObservation {
                timestamp: start_time + (i as i64) * 86400,
                symbol: symbol.to_string(),
                open,
                high,
                low,
                close: close_price,
                volume,
                vwap,
                spread,
            });

            current_price = close_price;
        }

        observations
    }

    /// Generates a synthetic dataset directly serialized into a columnar [`TabularDataset`].
    pub fn generate_tabular(
        &self,
        symbol: &str,
        count: usize,
    ) -> Result<TabularDataset, ParquetError> {
        let ticks = self.generate_ticks(symbol, count, 100.0, 0.20);
        FinancialObservation::to_tabular_dataset(&ticks)
    }
}

/// Procedural acoustic and audio signal telemetry generator for audio testing and synthesis verification.
#[derive(Debug, Clone)]
pub struct SyntheticAcousticGenerator {
    seed: u64,
}

impl SyntheticAcousticGenerator {
    /// Creates a new acoustic generator with the specified deterministic seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generates a series of realistic [`AcousticTelemetrySnapshot`] records.
    pub fn generate_telemetry(&self, count: usize) -> Vec<AcousticTelemetrySnapshot> {
        let mut prng = SimplePrng::new(self.seed);
        let mut snapshots = Vec::with_capacity(count);
        let start_time = 1700000000i64;

        for i in 0..count {
            let rms_db = -30.0 + prng.next_gaussian() * 6.0;
            let peak_db = rms_db + 6.0 + prng.next_f64() * 4.0;
            let centroid = 1500.0 + prng.next_gaussian() * 300.0;
            let zcr = (0.05 + prng.next_f64() * 0.15).clamp(0.0, 1.0);

            snapshots.push(AcousticTelemetrySnapshot {
                timestamp: start_time + (i as i64) * 10,
                frame_idx: i as i64,
                rms_db: rms_db.clamp(-90.0, 0.0),
                peak_db: peak_db.clamp(-90.0, 3.0),
                spectral_centroid: centroid.max(50.0),
                zero_crossing_rate: zcr,
                active_voices: (8.0 + prng.next_gaussian() * 2.0).round().max(1.0) as i64,
                buffer_underruns: 0,
            });
        }

        snapshots
    }

    /// Generates a synthetic frequency chirp sweep from $f_0$ to $f_1$ over `duration_secs`.
    pub fn generate_chirp(
        &self,
        sample_rate: u32,
        duration_secs: f32,
        f_start: f32,
        f_end: f32,
    ) -> Vec<f32> {
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let mut samples = Vec::with_capacity(num_samples);
        let t_end = duration_secs as f64;
        let k = (f_end as f64 - f_start as f64) / t_end;

        for i in 0..num_samples {
            let t = (i as f64) / (sample_rate as f64);
            let phase = 2.0 * std::f64::consts::PI * (f_start as f64 * t + 0.5 * k * t * t);
            samples.push(phase.sin() as f32);
        }

        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_market_generator() {
        let generator = SyntheticMarketGenerator::new(42);
        let ticks = generator.generate_ticks("BHP.AX", 100, 45.0, 0.18);
        assert_eq!(ticks.len(), 100);
        assert_eq!(ticks[0].symbol, "BHP.AX");
        assert!(ticks[0].close > 0.0);

        let dataset = FinancialObservation::to_tabular_dataset(&ticks).unwrap();
        assert_eq!(dataset.num_rows, 100);
        let restored = FinancialObservation::from_tabular_dataset(&dataset).unwrap();
        assert_eq!(restored.len(), 100);
        assert_eq!(restored[0].symbol, "BHP.AX");
    }

    #[test]
    fn test_synthetic_acoustic_generator() {
        let acoustic_gen = SyntheticAcousticGenerator::new(1337);
        let tele = acoustic_gen.generate_telemetry(50);
        assert_eq!(tele.len(), 50);
        assert!(tele[0].rms_db < 0.0);

        let chirp = acoustic_gen.generate_chirp(44100, 0.1, 200.0, 2000.0);
        assert_eq!(chirp.len(), 4410);
        assert!(chirp.iter().all(|&s| s.abs() <= 1.0));
    }
}
