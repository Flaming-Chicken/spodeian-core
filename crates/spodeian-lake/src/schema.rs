//! Canonical time-series and telemetry schemas for cross-workspace integration.
//!
//! Provides strongly-typed schemas and lossless bidirectional conversions
//! to [`TabularDataset`] for Parquet serialization and streaming.
//!
//! **Stability: Experimental** — API subject to change

use crate::parquet_io::{ParquetError, TabularDataset};
use serde::{Deserialize, Serialize};

/// Canonical financial observation recording multi-factor market ticks or bars.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinancialObservation {
    pub timestamp: i64,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: f64,
    pub spread: f64,
}

impl FinancialObservation {
    /// Converts a slice of [`FinancialObservation`] records into a columnar [`TabularDataset`].
    pub fn to_tabular_dataset(records: &[Self]) -> Result<TabularDataset, ParquetError> {
        let mut timestamps = Vec::with_capacity(records.len());
        let mut symbols = Vec::with_capacity(records.len());
        let mut opens = Vec::with_capacity(records.len());
        let mut highs = Vec::with_capacity(records.len());
        let mut lows = Vec::with_capacity(records.len());
        let mut closes = Vec::with_capacity(records.len());
        let mut volumes = Vec::with_capacity(records.len());
        let mut vwaps = Vec::with_capacity(records.len());
        let mut spreads = Vec::with_capacity(records.len());

        for rec in records {
            timestamps.push(rec.timestamp);
            symbols.push(rec.symbol.clone());
            opens.push(rec.open);
            highs.push(rec.high);
            lows.push(rec.low);
            closes.push(rec.close);
            volumes.push(rec.volume);
            vwaps.push(rec.vwap);
            spreads.push(rec.spread);
        }

        let mut dataset = TabularDataset::new();
        if records.is_empty() {
            dataset.column_names = vec![
                "timestamp".to_string(),
                "symbol".to_string(),
                "open".to_string(),
                "high".to_string(),
                "low".to_string(),
                "close".to_string(),
                "volume".to_string(),
                "vwap".to_string(),
                "spread".to_string(),
            ];
            return Ok(dataset);
        }

        dataset.add_int_column("timestamp", timestamps)?;
        dataset.add_string_column("symbol", symbols)?;
        dataset.add_float_column("open", opens)?;
        dataset.add_float_column("high", highs)?;
        dataset.add_float_column("low", lows)?;
        dataset.add_float_column("close", closes)?;
        dataset.add_float_column("volume", volumes)?;
        dataset.add_float_column("vwap", vwaps)?;
        dataset.add_float_column("spread", spreads)?;

        Ok(dataset)
    }

    /// Reconstructs [`FinancialObservation`] records from a [`TabularDataset`].
    pub fn from_tabular_dataset(dataset: &TabularDataset) -> Result<Vec<Self>, ParquetError> {
        if dataset.num_rows == 0 {
            return Ok(Vec::new());
        }

        let timestamps = dataset
            .int_columns
            .get("timestamp")
            .ok_or_else(|| ParquetError::ColumnNotFound("timestamp".to_string()))?;
        let symbols = dataset
            .string_columns
            .get("symbol")
            .ok_or_else(|| ParquetError::ColumnNotFound("symbol".to_string()))?;
        let opens = dataset
            .float_columns
            .get("open")
            .ok_or_else(|| ParquetError::ColumnNotFound("open".to_string()))?;
        let highs = dataset
            .float_columns
            .get("high")
            .ok_or_else(|| ParquetError::ColumnNotFound("high".to_string()))?;
        let lows = dataset
            .float_columns
            .get("low")
            .ok_or_else(|| ParquetError::ColumnNotFound("low".to_string()))?;
        let closes = dataset
            .float_columns
            .get("close")
            .ok_or_else(|| ParquetError::ColumnNotFound("close".to_string()))?;
        let volumes = dataset
            .float_columns
            .get("volume")
            .ok_or_else(|| ParquetError::ColumnNotFound("volume".to_string()))?;
        let vwaps = dataset
            .float_columns
            .get("vwap")
            .ok_or_else(|| ParquetError::ColumnNotFound("vwap".to_string()))?;
        let spreads = dataset
            .float_columns
            .get("spread")
            .ok_or_else(|| ParquetError::ColumnNotFound("spread".to_string()))?;

        let mut observations = Vec::with_capacity(dataset.num_rows);
        for i in 0..dataset.num_rows {
            observations.push(Self {
                timestamp: timestamps[i],
                symbol: symbols[i].clone(),
                open: opens[i],
                high: highs[i],
                low: lows[i],
                close: closes[i],
                volume: volumes[i],
                vwap: vwaps[i],
                spread: spreads[i],
            });
        }

        Ok(observations)
    }
}

/// Canonical acoustic telemetry snapshot recording real-time audio engine metrics.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticTelemetrySnapshot {
    pub timestamp: i64,
    pub frame_idx: i64,
    pub rms_db: f64,
    pub peak_db: f64,
    pub spectral_centroid: f64,
    pub zero_crossing_rate: f64,
    pub active_voices: i64,
    pub buffer_underruns: i64,
}

impl AcousticTelemetrySnapshot {
    /// Converts a slice of [`AcousticTelemetrySnapshot`] records into a columnar [`TabularDataset`].
    pub fn to_tabular_dataset(records: &[Self]) -> Result<TabularDataset, ParquetError> {
        let mut timestamps = Vec::with_capacity(records.len());
        let mut frame_indices = Vec::with_capacity(records.len());
        let mut rms_dbs = Vec::with_capacity(records.len());
        let mut peak_dbs = Vec::with_capacity(records.len());
        let mut centroids = Vec::with_capacity(records.len());
        let mut zcrs = Vec::with_capacity(records.len());
        let mut active_voices_vec = Vec::with_capacity(records.len());
        let mut buffer_underruns_vec = Vec::with_capacity(records.len());

        for rec in records {
            timestamps.push(rec.timestamp);
            frame_indices.push(rec.frame_idx);
            rms_dbs.push(rec.rms_db);
            peak_dbs.push(rec.peak_db);
            centroids.push(rec.spectral_centroid);
            zcrs.push(rec.zero_crossing_rate);
            active_voices_vec.push(rec.active_voices);
            buffer_underruns_vec.push(rec.buffer_underruns);
        }

        let mut dataset = TabularDataset::new();
        if records.is_empty() {
            dataset.column_names = vec![
                "timestamp".to_string(),
                "frame_idx".to_string(),
                "rms_db".to_string(),
                "peak_db".to_string(),
                "spectral_centroid".to_string(),
                "zero_crossing_rate".to_string(),
                "active_voices".to_string(),
                "buffer_underruns".to_string(),
            ];
            return Ok(dataset);
        }

        dataset.add_int_column("timestamp", timestamps)?;
        dataset.add_int_column("frame_idx", frame_indices)?;
        dataset.add_float_column("rms_db", rms_dbs)?;
        dataset.add_float_column("peak_db", peak_dbs)?;
        dataset.add_float_column("spectral_centroid", centroids)?;
        dataset.add_float_column("zero_crossing_rate", zcrs)?;
        dataset.add_int_column("active_voices", active_voices_vec)?;
        dataset.add_int_column("buffer_underruns", buffer_underruns_vec)?;

        Ok(dataset)
    }

    /// Reconstructs [`AcousticTelemetrySnapshot`] records from a [`TabularDataset`].
    pub fn from_tabular_dataset(dataset: &TabularDataset) -> Result<Vec<Self>, ParquetError> {
        if dataset.num_rows == 0 {
            return Ok(Vec::new());
        }

        let timestamps = dataset
            .int_columns
            .get("timestamp")
            .ok_or_else(|| ParquetError::ColumnNotFound("timestamp".to_string()))?;
        let frame_indices = dataset
            .int_columns
            .get("frame_idx")
            .ok_or_else(|| ParquetError::ColumnNotFound("frame_idx".to_string()))?;
        let rms_dbs = dataset
            .float_columns
            .get("rms_db")
            .ok_or_else(|| ParquetError::ColumnNotFound("rms_db".to_string()))?;
        let peak_dbs = dataset
            .float_columns
            .get("peak_db")
            .ok_or_else(|| ParquetError::ColumnNotFound("peak_db".to_string()))?;
        let centroids = dataset
            .float_columns
            .get("spectral_centroid")
            .ok_or_else(|| ParquetError::ColumnNotFound("spectral_centroid".to_string()))?;
        let zcrs = dataset
            .float_columns
            .get("zero_crossing_rate")
            .ok_or_else(|| ParquetError::ColumnNotFound("zero_crossing_rate".to_string()))?;
        let active_voices = dataset
            .int_columns
            .get("active_voices")
            .ok_or_else(|| ParquetError::ColumnNotFound("active_voices".to_string()))?;
        let buffer_underruns = dataset
            .int_columns
            .get("buffer_underruns")
            .ok_or_else(|| ParquetError::ColumnNotFound("buffer_underruns".to_string()))?;

        let mut snapshots = Vec::with_capacity(dataset.num_rows);
        for i in 0..dataset.num_rows {
            snapshots.push(Self {
                timestamp: timestamps[i],
                frame_idx: frame_indices[i],
                rms_db: rms_dbs[i],
                peak_db: peak_dbs[i],
                spectral_centroid: centroids[i],
                zero_crossing_rate: zcrs[i],
                active_voices: active_voices[i],
                buffer_underruns: buffer_underruns[i],
            });
        }

        Ok(snapshots)
    }
}

/// Canonical player progression and cultivation telemetry event.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlayerProgressEvent {
    pub timestamp: i64,
    pub session_id: String,
    pub realm_level: i64,
    pub void_resonance: f64,
    pub soul_shards: i64,
    pub mana_ratio: f64,
    pub event_type: String,
}

impl PlayerProgressEvent {
    /// Converts a slice of [`PlayerProgressEvent`] records into a columnar [`TabularDataset`].
    pub fn to_tabular_dataset(records: &[Self]) -> Result<TabularDataset, ParquetError> {
        let mut timestamps = Vec::with_capacity(records.len());
        let mut session_ids = Vec::with_capacity(records.len());
        let mut realm_levels = Vec::with_capacity(records.len());
        let mut void_resonances = Vec::with_capacity(records.len());
        let mut soul_shards_vec = Vec::with_capacity(records.len());
        let mut mana_ratios = Vec::with_capacity(records.len());
        let mut event_types = Vec::with_capacity(records.len());

        for rec in records {
            timestamps.push(rec.timestamp);
            session_ids.push(rec.session_id.clone());
            realm_levels.push(rec.realm_level);
            void_resonances.push(rec.void_resonance);
            soul_shards_vec.push(rec.soul_shards);
            mana_ratios.push(rec.mana_ratio);
            event_types.push(rec.event_type.clone());
        }

        let mut dataset = TabularDataset::new();
        if records.is_empty() {
            dataset.column_names = vec![
                "timestamp".to_string(),
                "session_id".to_string(),
                "realm_level".to_string(),
                "void_resonance".to_string(),
                "soul_shards".to_string(),
                "mana_ratio".to_string(),
                "event_type".to_string(),
            ];
            return Ok(dataset);
        }

        dataset.add_int_column("timestamp", timestamps)?;
        dataset.add_string_column("session_id", session_ids)?;
        dataset.add_int_column("realm_level", realm_levels)?;
        dataset.add_float_column("void_resonance", void_resonances)?;
        dataset.add_int_column("soul_shards", soul_shards_vec)?;
        dataset.add_float_column("mana_ratio", mana_ratios)?;
        dataset.add_string_column("event_type", event_types)?;

        Ok(dataset)
    }

    /// Reconstructs [`PlayerProgressEvent`] records from a [`TabularDataset`].
    pub fn from_tabular_dataset(dataset: &TabularDataset) -> Result<Vec<Self>, ParquetError> {
        if dataset.num_rows == 0 {
            return Ok(Vec::new());
        }

        let timestamps = dataset
            .int_columns
            .get("timestamp")
            .ok_or_else(|| ParquetError::ColumnNotFound("timestamp".to_string()))?;
        let session_ids = dataset
            .string_columns
            .get("session_id")
            .ok_or_else(|| ParquetError::ColumnNotFound("session_id".to_string()))?;
        let realm_levels = dataset
            .int_columns
            .get("realm_level")
            .ok_or_else(|| ParquetError::ColumnNotFound("realm_level".to_string()))?;
        let void_resonances = dataset
            .float_columns
            .get("void_resonance")
            .ok_or_else(|| ParquetError::ColumnNotFound("void_resonance".to_string()))?;
        let soul_shards = dataset
            .int_columns
            .get("soul_shards")
            .ok_or_else(|| ParquetError::ColumnNotFound("soul_shards".to_string()))?;
        let mana_ratios = dataset
            .float_columns
            .get("mana_ratio")
            .ok_or_else(|| ParquetError::ColumnNotFound("mana_ratio".to_string()))?;
        let event_types = dataset
            .string_columns
            .get("event_type")
            .ok_or_else(|| ParquetError::ColumnNotFound("event_type".to_string()))?;

        let mut events = Vec::with_capacity(dataset.num_rows);
        for i in 0..dataset.num_rows {
            events.push(Self {
                timestamp: timestamps[i],
                session_id: session_ids[i].clone(),
                realm_level: realm_levels[i],
                void_resonance: void_resonances[i],
                soul_shards: soul_shards[i],
                mana_ratio: mana_ratios[i],
                event_type: event_types[i].clone(),
            });
        }

        Ok(events)
    }
}
