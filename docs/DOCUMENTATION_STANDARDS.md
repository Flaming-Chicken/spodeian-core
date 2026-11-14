# Spodeian Workspace Documentation Standards

**Cross-Crate Rust, Python & Mathematical Documentation Conventions**

---

## 1. Overview & Core Philosophy

`spodeian-core` provides the shared serialization, UI, persistence, caching, parallelism, telemetry, and scientific machine learning primitives across all Spodeian projects.

### The Four Pillars:
1. **Stability Guarantees**: Public crates (`spodeian-parallel`, `spodeian-cache`, `spodeian-export`, `spodeian-ui`, `spodeian-web-utils`, `spodeian-telemetry`, `spodeian-tokens`, `spodeian-core`) explicitly mark stability tiers in module and symbol docstrings.
2. **Mathematical & Algorithmic Transparency**: Every optimizer, meta-learning rule, and numerical kernel must state its mathematical formulation, complexity, and failure modes in KaTeX notation.
3. **Actionable Examples**: Provide self-contained, compile-tested doctests (`# Examples`) demonstrating standard usage and edge cases.
4. **Zero-Warning Hygiene**: `cargo doc --no-deps --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` must compile cleanly without warnings or broken intra-doc links.

---

## 2. KaTeX Mathematical Typesetting Conventions

- **Inline Math**: Enclose mathematical expressions between single dollar signs: `$f(x) = \sin(x)$`.
- **Display Math**: Place multi-line equations inside double dollar signs on their own lines (`$$...$$`).
- **Rustdoc Escaping**: Escape brackets inside KaTeX expressions (`\[n\]`) so rustdoc does not interpret them as intra-doc links.
- **Vectors & Matrices**: Use bold Roman `\mathbf{v}, \mathbf{A}, \mathbf{x}`.
- **Operators**: Use `\operatorname{...}` for named functions.

---

## 3. Verification Checklist

- [ ] `cargo doc --no-deps --workspace` completes with **0 warnings** and **0 broken links**.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes with **0 warnings**.
- [ ] `cargo test --workspace` passes 100% of unit and integration tests.
