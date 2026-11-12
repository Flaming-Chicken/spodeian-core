# arXiv Preprint Package: `SelfOptimizingMeta` & `SelfOptimizingMetaIntegrator`

This directory contains the complete, self-contained **LuaLaTeX + BibLaTeX (`style=ieee`)** manuscript for arXiv release (`math.OC`, `cs.LG`, `math.NA`):

- **`self_optimizing_meta_arxiv.tex`**: Main LuaLaTeX manuscript (with embedded `filecontents*` for single-file portability, formal theorems/propositions/proofs, `algorithm2e`/`algpseudocode` blocks, native TikZ/PGFPlots vector figures, and 5 `booktabs` empirical benchmark tables).
- **`references.bib`**: Standalone BibLaTeX (`biber`) IEEE bibliography database containing 22 peer-reviewed references with DOIs and arXiv eprint identifiers.
- **`latexmkrc`**: Automated build configuration specifying LuaLaTeX (`$pdf_mode = 4`) and `biber`.

## Compilation Instructions

### Option 1: Using `latexmk` (Recommended for TeX Live / Overleaf / arXiv)
```bash
latexmk -lualatex self_optimizing_meta_arxiv.tex
```

### Option 2: Direct `lualatex` + `biber` Sequence
```bash
lualatex -interaction=nonstopmode self_optimizing_meta_arxiv.tex
biber self_optimizing_meta_arxiv
lualatex -interaction=nonstopmode self_optimizing_meta_arxiv.tex
lualatex -interaction=nonstopmode self_optimizing_meta_arxiv.tex
```

### arXiv Submission Notes
When uploading to arXiv (which uses TeX Live with LuaLaTeX and `biber` support):
1. Include `self_optimizing_meta_arxiv.tex`, `references.bib`, and the generated `self_optimizing_meta_arxiv.bbl` (matching the arXiv TeX Live `biblatex`/`biber` version).
2. The magic header `% !TeX program = lualatex` and `\RequirePackage{luatex85}` at the top of `self_optimizing_meta_arxiv.tex` instruct modern TeX engines and build systems to invoke `lualatex`.
