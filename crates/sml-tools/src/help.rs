//! Detailed help documentation and shell completions for the `aix` CLI toolchain.

pub const USAGE: &str = "\
aix — AI-Examples repository toolchain (README.md is the shared plan + contract)

USAGE: cargo aix <command> [options]      (or: python scripts/verify_repo.py <command> ...)

COMMANDS
  check          Validate repository structure, contracts, links, and weights.
                   --only a,b      subset of: structure,notebooks,markdown,links,forbidden,weights
                   --path P, -p    limit file-level checks to a path (repeatable)
                   --topic T, -t   limit file-level checks to one topic's track files
                   --semantics, -s enforce notebook Cell 1–8 semantic rules from contract
                   --fix, -f       rewrite links with an unambiguous fix (wrong depth, stale slug, case)
                   --format F      human (default) | json | github
                   --max N         diagnostics shown per check (default: 25)
                   --quiet, -q     suppress passing check sections; only show warnings/errors
  sync           Regenerate lib.rs module trees, candle test aggregators, and README status/progress.
                   --check         fail instead of writing when anything is out of date (CI gate)
  status         Curriculum progress summary across all 18 categories.
                   --wip           list in-progress [~] topics and missing tracks
                   --category C, -c focus summary on a specific category
                   --format F      human (default) | json
  next           Sequential single-active-topic resolver.
                   --category C, -c filter next topic to a specific category
                   --format F      human (default) | json
  scaffold T     Create stubs for curriculum topic T in every track + candle test, then sync.
                   --dry-run       show files that would be created without writing to disk
                   --force         overwrite existing track files
  notebook T     8-cell notebook compiler, exporter, and contract inspector.
                   --from FILE     compile percent-format FILE (# %% [markdown] / # %%) into .ipynb
                   --export FILE   export notebooks/<id>.ipynb to percent-format FILE (or - for stdout)
                   --check         validate 8-cell layout, empty outputs, and semantic rules
  weights        SafeTensors header inspector, golden exporter, and tensor diff tool.
                   inspect FILE... parse binary SafeTensors headers (metadata, shapes, dtypes)
                   export T        regenerate T's golden weights via python pytorch2/<id>.py --export-golden
                   diff A B        compare two SafeTensors files tensor-by-tensor (byte & F32 diff)
                   check           verify golden weights exist and contain required tensors
                   regen [options] batch-regenerate golden weights (--topic T | --path P, --check)
                   <topic_or_file> auto-inspect SafeTensors header of file or curriculum topic
  run-notebooks  Execute notebooks, one isolated Python process each, hard-killed on timeout.
                   --path P, -p | --topic T, -t
                   --jobs N, -j    worker processes (default: min(cpus, 8))
                   --budget S      notebook CPU budget in seconds (default: 3.0 from contract)
                   --grace S       startup allowance before hard kill (default: 45)
                   --python PY     explicit Python interpreter path
                   --verbose, -v   print worker output even for passing notebooks
  test           Topic inner loop or crate test suites.
                   --topic T, -t   cargo tests (both crates) filtered to T + notebook + pytorch2 + jax
                   --crate C       run `cargo test -p C --tests` (repeatable; default: both)
                   --jobs N, -j    cargo -j
                   --python PY     explicit Python interpreter path
                   --verbose, -v   print detailed test output
  ci             check --semantics + run-notebooks + workspace crate test suites.
                   --format F      human (default) | json | github
                   --jobs N, -j    cargo/worker parallelism
                   --python PY     explicit Python interpreter path
  doctor         Validate environment, toolchain, Python packages, and repository health.
                   --python PY     explicit Python interpreter path
  clean          Remove temporary target/aix timings, __pycache__, and .ipynb_checkpoints.
                   --dry-run       show what would be removed without deleting
  completions    Generate shell completion script (bash | zsh | fish | powershell).
  verify         verify_repo.py legacy compatibility layer.
  help [cmd]     Show global help or help for a specific command (e.g. `cargo aix help check`).
";

pub const CHECK_HELP: &str = "\
aix check — Validate repository structure, contracts, links, and weights

USAGE: cargo aix check [options]

OPTIONS:
  --only <checks>     Comma-separated subset of checks to run:
                      structure, notebooks, markdown, links, forbidden, weights
  --path <path>, -p   Limit checks to files matching this path prefix (repeatable)
  --topic <id>, -t    Limit checks to track files of this curriculum topic
  --semantics, -s     Enforce notebook Cell 1–8 semantic rules from contract
  --fix, -f           Auto-repair broken links (depth errors, stale slugs, case)
  --format <fmt>      Output format: human (default) | json | github
  --max <N>           Maximum diagnostics to display per check section (default: 25)
  --quiet, -q         Suppress passing sections; only show warnings and errors
  --help, -h          Show this help message

CHECKS EXECUTED:
  1. structure:  Verifies 1-to-1 track parity against 263 curriculum topics in README.md,
                 detects checklist status drift, verifies generated module tree freshness,
                 and enforces scripts/ directory hygiene.
  2. notebooks:  Verifies exact 8-cell layout (markdown, markdown, code, code, code, code, code, markdown),
                 empty outputs, stripped execution counts, and line budgets.
  3. markdown:   Strictly validates 8 required architectural sections when contract enforce is true.
  4. links:      Extracts and validates all Markdown links across .md and .ipynb files;
                 detects case variants for Linux CI compatibility; auto-repairs with --fix.
  5. forbidden:  Scans all 6 code tracks for unmocked network calls (requests, urllib, curl, socket).
  6. weights:    Parses SafeTensors headers and validates required tensor groups.

EXAMPLES:
  cargo aix check
  cargo aix check --semantics
  cargo aix check --only links --fix
  cargo aix check --topic 00_foundations/01_linear_neuron_and_invertible_transforms
  cargo aix check --format github
";

pub const SYNC_HELP: &str = "\
aix sync — Regenerate derived module trees, test aggregators, and progress tables

USAGE: cargo aix sync [options]

OPTIONS:
  --check             Check if all generated files are up to date; exit with code 1
                      if anything is stale (used in CI)
  --help, -h          Show this help message

AFFECTED TARGETS:
  - crates/ai-candle/src/lib.rs        (Rust Candle topic modules inside <aix:generated>)
  - crates/ai-kernel/src/lib.rs        (Rust WGPU/CPU kernel modules inside <aix:generated>)
  - crates/ai-candle/tests/<cat>.rs    (Integration test runner modules)
  - README.md                          (Curriculum [x]/[~]/[ ] marks and progress table)
  - README.md & INDEX_BY_GOAL.md       (Top-of-file Progress-XX/263 badges)

EXAMPLES:
  cargo aix sync
  cargo aix sync --check
";

pub const STATUS_HELP: &str = "\
aix status — Curriculum progress summary across all 18 categories

USAGE: cargo aix status [options]

OPTIONS:
  --wip               List all in-progress [~] topics and their missing track files
  --category <cat>, -c Focus summary on a specific category prefix
  --format <fmt>      Output format: human (default) | json
  --help, -h          Show this help message

EXAMPLES:
  cargo aix status
  cargo aix status --wip
  cargo aix status -c 05_architectures
  cargo aix status --format json
";

pub const NEXT_HELP: &str = "\
aix next — Sequential single-active-topic resolver

USAGE: cargo aix next [options]

OPTIONS:
  --category <cat>, -c Filter next topic to a specific category
  --format <fmt>      Output format: human (default) | json
  --help, -h          Show this help message

BEHAVIOR:
  Returns the single active [~] WIP topic and its missing tracks.
  If no topic is currently WIP, returns the next [ ] planned topic in sequential curriculum order.
  This enforces the sequential single-active-topic pair programming protocol.

EXAMPLES:
  cargo aix next
  cargo aix next --cat 05_architectures
  cargo aix next --format json
";

pub const SCAFFOLD_HELP: &str = "\
aix scaffold — Generate contract-conforming 7-track stubs for a topic

USAGE: cargo aix scaffold <topic> [options]

ARGUMENTS:
  <topic>             Topic ID, slug, or number from README.md curriculum

OPTIONS:
  --dry-run           Print files that would be created without writing to disk
  --force             Overwrite existing files (default: skip existing files)
  --help, -h          Show this help message

GENERATED TRACKS:
  - markdown/<id>.md                   (Track 1: Architectural reference with 8 sections)
  - notebooks/<id>.ipynb               (Track 2: Interactive concept notebook, 8 cells)
  - pytorch2/<id>.py                   (Track 3: PyTorch 2 module + --export-golden CLI)
  - jax/<id>.py                        (Track 4: Functional JAX/Equinox module)
  - candle/<id>.rs                     (Track 5: Native Rust Candle module)
  - kernels/<id>.rs                    (Companion A: Native Rust WGPU/CPU kernel)
  - crates/ai-candle/tests/<cat>/...   (Integration test stub)
  Automatically runs `sync` after file creation to register all Rust modules.

EXAMPLES:
  cargo aix scaffold 05_architectures/03_recurrent_ssm_and_memory/02_xlstm_slstm_and_mlstm
  cargo aix scaffold 05_03_02 --dry-run
";

pub const NOTEBOOK_HELP: &str = "\
aix notebook — 8-cell notebook compiler, exporter, and contract inspector

USAGE: cargo aix notebook <topic> [options]

ARGUMENTS:
  <topic>             Topic ID, slug, or number from README.md curriculum

OPTIONS:
  --from <file.py>    Compile percent-format source (# %% [markdown] / # %%)
                      into notebooks/<id>.ipynb (stripped of outputs)
  --export <file.py>  Export notebooks/<id>.ipynb to percent-format source
                      (use '-' for stdout)
  --check             Validate notebook cell count, types, stripped outputs,
                      and semantic contract rules
  --help, -h          Show this help message

DEFAULT BEHAVIOR:
  When invoked without --from or --export, prints a cell-by-cell inspection
  summary showing cell types, line counts, output stripping status, and contract compliance.

EXAMPLES:
  cargo aix notebook 00_foundations/01_linear_neuron_and_invertible_transforms
  cargo aix notebook 00_01 --export my_nb.py
  cargo aix notebook 00_01 --from my_nb.py
  cargo aix notebook 00_01 --check
";

pub const WEIGHTS_HELP: &str = "\
aix weights — SafeTensors header inspector, golden exporter, and tensor diff tool

USAGE:
  cargo aix weights inspect <file.safetensors...>
  cargo aix weights export <topic>
  cargo aix weights diff <file_a> <file_b>
  cargo aix weights check [--topic <T>] [--path <P>]
  cargo aix weights regen [--topic <T>] [--path <P>] [--check]
  cargo aix weights <topic_or_file>

COMMANDS:
  inspect <files...>  Parse binary SafeTensors headers directly in Rust (no PyTorch);
                      display metadata, tensor names, dtypes, shapes, byte sizes.
  export <topic>      Regenerate topic's golden weights via:
                      python pytorch2/<id>.py --export-golden
  diff <A> <B>        Compare two SafeTensors files tensor-by-tensor directly in Rust;
                      reports additions, removals, shape/dtype mismatches,
                      byte-identical tensors, and max absolute F32 error.
  check               Verify that golden weights exist and contain all required tensors.
  regen               Batch-regenerate golden weights via --export-golden.
                      Use --check to verify deterministic bitwise reproducibility.
  <topic_or_file>     Direct shorthand: inspects SafeTensors header if file or topic ID is given.

OPTIONS:
  --topic <T>, -t     Limit operation to a specific topic
  --path <P>, -p      Limit operation to a path prefix
  --check             Verify reproducibility or required tensors
  --python <path>     Explicit Python interpreter path
  --help, -h          Show this help message

EXAMPLES:
  cargo aix weights inspect weights/00_foundations/01_linear_neuron_and_invertible_transforms.safetensors
  cargo aix weights export 00_foundations/01_linear_neuron_and_invertible_transforms
  cargo aix weights diff weights/a.safetensors weights/b.safetensors
  cargo aix weights check
  cargo aix weights 00_01
";

pub const RUN_NOTEBOOKS_HELP: &str = "\
aix run-notebooks — Process-isolated parallel notebook runner

USAGE: cargo aix run-notebooks [options]

OPTIONS:
  --path <P>, -p      Limit execution to notebooks matching path prefix
  --topic <T>, -t     Limit execution to a single topic's notebook
  --jobs <N>, -j      Number of parallel worker processes (default: min(cpus, 8))
  --budget <secs>     CPU time budget per notebook (default: 3.0s from contract)
  --grace <secs>      Startup & import allowance before hard OS kill (default: 45s)
  --python <path>     Explicit Python interpreter path
  --verbose, -v       Print worker stdout and stderr even for passing notebooks
  --help, -h          Show this help message

BEHAVIOR:
  Each notebook runs in an isolated Python process (scripts/aix_nb_worker.py)
  with a hard OS-level wall-clock termination deadline (budget + grace).
  Outputs run timings to target/aix/notebook_timings.json.

EXAMPLES:
  cargo aix run-notebooks --topic 00_01
  cargo aix run-notebooks --jobs 4 --budget 3.0
";

pub const TEST_HELP: &str = "\
aix test — Topic inner-loop runner or workspace crate test suite

USAGE: cargo aix test [options]

OPTIONS:
  --topic <T>, -t     Verify all implementations of a single topic:
                      1. cargo test -p ai-candle --tests -- <filter>
                      2. cargo test -p ai-kernel --tests -- <filter>
                      3. python pytorch2/<id>.py (parity vs golden weights)
                      4. python jax/<id>.py (parity vs golden weights)
                      5. run-notebooks for notebooks/<id>.ipynb
                      6. scoped aix check --semantics
  --crate <C>         Run cargo test -p <C> --tests (repeatable; default: both)
  --jobs <N>, -j      Pass -j <N> to cargo test
  --python <path>     Explicit Python interpreter path
  --verbose, -v       Print detailed output
  --help, -h          Show this help message

EXAMPLES:
  cargo aix test --topic 00_foundations/01_linear_neuron_and_invertible_transforms
  cargo aix test --crate ai-candle
";

pub const CI_HELP: &str = "\
aix ci — Complete repository continuous integration gate

USAGE: cargo aix ci [options]

OPTIONS:
  --format <fmt>      human (default) | json | github (workflow annotations)
  --jobs <N>, -j      Parallelism for notebook and cargo test execution
  --python <path>     Explicit Python interpreter path
  --help, -h          Show this help message

CHECKS EXECUTED:
  1. aix check --semantics (structure, notebooks, markdown, links, forbidden, weights)
  2. aix run-notebooks (all notebooks within budget)
  3. cargo test --workspace --tests (ai-architect-schema, aix, ai-candle, ai-kernel)

EXAMPLES:
  cargo aix ci
  cargo aix ci --format github
";

pub const DOCTOR_HELP: &str = "\
aix doctor — Repository, toolchain, and environment diagnostics

USAGE: cargo aix doctor [options]

OPTIONS:
  --python <path>     Explicit Python interpreter to evaluate
  --help, -h          Show this help message

DIAGNOSTICS PERFORMED:
  - Repository root, curriculum size (263), and contract validity
  - Overall progress (done, wip, planned, percentage)
  - Rust toolchain and cargo availability
  - Workspace crates existence (ai-architect-schema, ai-candle, ai-kernel, aix)
  - Python interpreter detection (probing .venv, venv, and system installations)
  - ML package availability (torch, safetensors, jax, matplotlib, numpy)
  - Full static check diagnostic status

EXAMPLES:
  cargo aix doctor
";

pub const CLEAN_HELP: &str = "\
aix clean — Remove temporary artifacts, caches, and test outputs

USAGE: cargo aix clean [options]

OPTIONS:
  --dry-run           Display items that would be removed without deleting
  --help, -h          Show this help message

TARGETS REMOVED:
  - target/aix/                   (Notebook execution timings and reports)
  - target/tmp/                   (Temporary build artifacts)
  - **/ __pycache__ /             (Python bytecode directories)
  - **/.ipynb_checkpoints/        (Jupyter checkpoint autosaves)
  - **/*.pyc, **/*.pyo            (Orphaned compiled Python files)

EXAMPLES:
  cargo aix clean --dry-run
  cargo aix clean
";

pub const COMPLETIONS_HELP: &str = "\
aix completions — Generate shell completion scripts

USAGE: cargo aix completions <shell>

ARGUMENTS:
  <shell>             bash | zsh | fish | powershell

EXAMPLES:
  # Bash:
  cargo aix completions bash > ~/.local/share/bash-completion/completions/aix

  # Zsh:
  cargo aix completions zsh > ~/.zfunc/_aix

  # Fish:
  cargo aix completions fish > ~/.config/fish/completions/aix.fish

  # PowerShell:
  cargo aix completions powershell >> $PROFILE
";

pub const VERIFY_HELP: &str = "\
aix verify — Legacy python scripts/verify_repo.py compatibility layer

USAGE: cargo aix verify [legacy_options]

SUPPORTED FLAGS:
  --check-symmetry    Check directory symmetry and contract compliance
  --check-links       Check for broken relative links
  --check-semantics   Check Cell 1–8 notebook semantic rules
  --check-candle      Run ai-candle integration tests
  --check-kernel      Run ai-kernel integration tests
  --run               Execute notebooks via worker processes
  --timeout <S>       Notebook CPU timeout in seconds
  --workers <N>       Parallel worker process count
  --parallel / --no-parallel
  --path <P>          Path filter
";

pub const POWERSHELL_COMPLETIONS: &str = r#"# PowerShell completion script for aix / cargo-aix
Register-ArgumentCompleter -Native -CommandName aix, cargo-aix -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $commands = @(
        'check', 'sync', 'status', 'next', 'scaffold', 'notebook',
        'weights', 'run-notebooks', 'test', 'verify', 'ci', 'doctor',
        'clean', 'completions', 'help'
    )
    $commands | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}
"#;

pub const BASH_COMPLETIONS: &str = r#"# Bash completion script for aix
_aix_completions() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local commands="check sync status next scaffold notebook weights run-notebooks test verify ci doctor clean completions help"
    COMPREPLY=( $(compgen -W "${commands}" -- "${cur}") )
}
complete -F _aix_completions aix
complete -F _aix_completions cargo-aix
"#;

pub const ZSH_COMPLETIONS: &str = r#"#compdef aix cargo-aix

_aix() {
    local -a commands
    commands=(
        'check:Validate repository structure, contracts, links, and weights'
        'sync:Regenerate module trees, test aggregators, and progress table'
        'status:Curriculum progress summary across all 18 categories'
        'next:Sequential single-active-topic resolver'
        'scaffold:Generate 7-track stubs for a topic'
        'notebook:8-cell notebook compiler, exporter, and inspector'
        'weights:SafeTensors header inspector and diff tool'
        'run-notebooks:Process-isolated parallel notebook runner'
        'test:Topic inner loop or workspace crate test suite'
        'ci:Complete continuous integration gate'
        'doctor:Repository, toolchain, and environment diagnostics'
        'clean:Remove temporary artifacts, caches, and test outputs'
        'completions:Generate shell completion script'
        'verify:Legacy verify_repo.py compatibility layer'
        'help:Show help message'
    )
    _describe 'command' commands
}

_aix "$@"
"#;

pub const FISH_COMPLETIONS: &str = r#"# Fish completion script for aix
complete -c aix -f -a "check sync status next scaffold notebook weights run-notebooks test verify ci doctor clean completions help"
complete -c cargo-aix -f -a "check sync status next scaffold notebook weights run-notebooks test verify ci doctor clean completions help"
"#;

pub fn print_cmd_help(cmd: &str) {
    match cmd {
        "check" => print!("{CHECK_HELP}"),
        "sync" => print!("{SYNC_HELP}"),
        "status" => print!("{STATUS_HELP}"),
        "next" => print!("{NEXT_HELP}"),
        "scaffold" => print!("{SCAFFOLD_HELP}"),
        "notebook" => print!("{NOTEBOOK_HELP}"),
        "weights" => print!("{WEIGHTS_HELP}"),
        "run-notebooks" => print!("{RUN_NOTEBOOKS_HELP}"),
        "test" => print!("{TEST_HELP}"),
        "ci" => print!("{CI_HELP}"),
        "doctor" => print!("{DOCTOR_HELP}"),
        "clean" => print!("{CLEAN_HELP}"),
        "completions" => print!("{COMPLETIONS_HELP}"),
        "verify" => print!("{VERIFY_HELP}"),
        _ => print!("{USAGE}"),
    }
}
