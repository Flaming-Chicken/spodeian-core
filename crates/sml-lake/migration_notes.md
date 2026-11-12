# `spodeian-lake` Migration Notes

**Stability Tier**: `Experimental (0.x)`  
**Current Version**: `0.1.0`

> [!NOTE]
> As an **Experimental** crate in `spodeian-core`, public APIs in `spodeian-lake` may change between minor versions (`0.1.0` -> `0.2.0`). All breaking modifications, schema adjustments, and migration paths will be catalogued in this document. Downstream projects must pin dependencies with caret requirements (`^0.1.0`).

---

## Version 0.1.0 (Initial Experimental Release)

### Core Capabilities
- **Local Windows Data Lake**: Managed storage root at `%LOCALAPPDATA%\spodeian\lake` with automatic directory structure (`data/`, `lake_registry.json`, `lake_registry.wal`).
- **Apache Parquet Storage**: Columnar storage for `TabularDataset` with `f64`, `i64`, and `String` columns.
- **SHA-256 Checksum Verification**: Cryptographic verification on write and read, with parallel multicore verification via Rayon (`verify_partitions_parallel`).
- **Dual-State WAL Architecture**: Crash-resilient Write-Ahead Log (`LakeWalEntry`) appending state mutations for immediate durability with snapshot compaction.
- **Value-Aware Retention & Eviction**: Multi-criteria priority pruning taking into account loss gradient feedback, rare crisis regime flags (`anomaly_score`), training saturation (`epochs_seen`), and hard crisis memory pinning (`is_pinned`).
- **Experiment Tracking Registry**: Unified registry for machine learning runs, factor benchmarks, and backtests (`ExperimentRecord`) storing configuration snapshots, metrics, hardware context, and checkpoint hashes.
- **Lineage & Citation Distiller**: Conversion of dataset metadata into BibTeX, CFF 1.2.0, APA, and Schema.org JSON-LD.

### Breaking Changes
- N/A (Initial release).

### Upgrade & Migration Guide
- Downstream projects should declare:
  ```toml
  spodeian-lake = { path = "../spodeian-core/crates/spodeian-lake", version = "^0.1.0" }
  ```
