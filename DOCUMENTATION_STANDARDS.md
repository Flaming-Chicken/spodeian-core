# Spodeian Core Documentation Standards

**Preferential Multi-Tier Caching, WCAG Design Tokens & Typography Conventions**

---

## 1. Overview & Core Philosophy

`spodeian-core` provides the shared infrastructure foundation for the entire Websites ecosystem. It standardizes multi-tier browser and native caching, WCAG 2.1 accessible color tokens, and KaTeX mathematical typography.

### The Five Pillars:
1. **Preferential Storage Mapping**:
   - **Cache API (`caches`)**: Large, immutable binary assets (`.safetensors`, `.wasm`, `.parquet`, audio buffers).
   - **IndexedDB (`idb`)**: Structured, relational, transaction-safe records (user states, ticks, factor loadings, replay memories).
   - **LocalStorage**: Fast, synchronous configuration toggles (< 16 KB).
   - **Native CAS**: Cryptographic Content-Addressed Storage on disk with SHA-256 validation.
2. **Deterministic Parity**: Exact identical hashing and serialization across platforms.
3. **Accessibility First**: All color palettes must mathematically guarantee WCAG 2.1 AA (contrast ratio $\ge 4.5:1$) or AAA ($\ge 7.0:1$).
4. **Zero-Warning Hygiene**: `cargo clippy --workspace --all-targets -- -D warnings` must compile with 0 warnings.
5. **Strict Test Isolation**: All tests reside in dedicated test files under `tests/`; no inline tests inside production source files.

---

## 2. Verification Checklist

Before opening PRs to `main`:
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes with 0 warnings.
- [ ] `cargo test --workspace` passes 100% of integration tests.
- [ ] Android mobile build (`scripts/build-android.ps1` / `scripts/build-android.sh`) verifies with `cargo-ndk`.
- [ ] No `DOCUMENTATION_STANDARDS.md` or internal roadmap files are included on `main`.
