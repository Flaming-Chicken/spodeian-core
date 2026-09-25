# Spodeian Core

**Cross-Project Shared Ecosystem Foundation: Preferential Multi-Tier Caching, WCAG 2.1 Design Tokens & Mathematical Typography**

---

## Architecture

- **`crates/spodeian-cache`**: Preferential storage engine routing large immutable binary payloads (`.safetensors`, `.wasm`, `.parquet`) to Cache API / Native CAS, and relational transaction records to IndexedDB.
- **`crates/spodeian-tokens`**: Shared theme palettes (Dark, Light, High-Contrast Dark/Light) guaranteed to meet WCAG 2.1 AA ($\ge 4.5:1$) and AAA ($\ge 7.0:1$) contrast standards, plus responsive screen breakpoints.
