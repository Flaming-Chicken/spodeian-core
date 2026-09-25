# Spodeian Core Development Roadmap

---

## Phase 1: Core Foundation [CURRENT]
- [x] Multi-crate workspace scaffold (`spodeian-cache`, `spodeian-tokens`).
- [x] Cryptographic SHA-256 Content-Addressed Storage (CAS) with tamper detection.
- [x] Preferential routing logic (Cache API vs. IndexedDB vs. LocalStorage).
- [x] WCAG 2.1 AA/AAA contrast ratios and responsive screen breakpoints.
- [x] Isolated integration test suite (`tests/`).
- [x] Universal `cargo-ndk` Android mobile deployment pipeline (`scripts/build-android.ps1`, `scripts/build-android.sh`).

---

## Phase 2: WASM Web Bindings & Service Worker Sync
- [ ] Browser Cache API and IndexedDB WASM async actor wrappers.
- [ ] Service worker background streaming cache coordinator.
- [ ] KaTeX math macro dynamic injector.
