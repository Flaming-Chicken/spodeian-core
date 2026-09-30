# Spodeian Core Release Automation (PowerShell)
# Usage: .\scripts\release.ps1 -Version "1.0.0" [-Push] [-DryRun]

param (
    [Parameter(Mandatory = $true, HelpMessage = "Semantic version to release (e.g., 1.0.1)")]
    [string]$Version,

    [switch]$Push,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "=== Spodeian Core Release Automation: v$Version ===" -ForegroundColor Cyan

# 1. Ensure we are in the repository root
$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot

# 2. Check git branch and status
$CurrentBranch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Host "Active Git Branch: $CurrentBranch" -ForegroundColor Yellow

$Status = git status --porcelain
if ($Status -and -not $DryRun) {
    Write-Host "Notice: Working directory contains uncommitted changes. These will be included in the release commit." -ForegroundColor Yellow
}

# 3. Bump version in root Cargo.toml if requested
$CargoTomlPath = Join-Path $RepoRoot "Cargo.toml"
$CargoToml = Get-Content $CargoTomlPath -Raw
if ($CargoToml -match '\[workspace\.package\][\s\S]*?version = "([^"]+)"') {
    $OldVersion = $Matches[1]
    Write-Host "Current Workspace Version: $OldVersion -> Target: $Version" -ForegroundColor Green
    if ($OldVersion -ne $Version) {
        $NewCargoToml = $CargoToml -replace '(\[workspace\.package\][\s\S]*?version = ")[^"]+(")', "`$1$Version`$2"
        Set-Content -Path $CargoTomlPath -Value $NewCargoToml -NoNewline
        Write-Host "Updated $CargoTomlPath" -ForegroundColor Green
    }
}

# 4. Run tests across workspace
Write-Host "`n[1/4] Running workspace tests..." -ForegroundColor Cyan
cargo test --workspace
if ($LASTEXITCODE -ne 0) {
    Write-Error "Workspace tests failed! Aborting release."
}

# 5. Run clippy lints across workspace
Write-Host "`n[2/4] Running workspace clippy check..." -ForegroundColor Cyan
cargo clippy --workspace -- -D warnings
if ($LASTEXITCODE -ne 0) {
    Write-Error "Clippy check failed with warnings! Aborting release."
}

# 6. Check semver breaking changes if cargo-semver-checks is available
Write-Host "`n[3/4] Checking semver contracts..." -ForegroundColor Cyan
$SemverCheckTool = Get-Command "cargo-semver-checks" -ErrorAction SilentlyContinue
if ($SemverCheckTool) {
    cargo semver-checks check-release
} else {
    Write-Host "cargo-semver-checks not installed in environment; skipping automated breaking change check." -ForegroundColor Yellow
    Write-Host "(Install with: cargo install cargo-semver-checks --locked)" -ForegroundColor DarkGray
}

# 7. Commit, Tag, and Push
$TagName = "spodeian-core-v$Version"
Write-Host "`n[4/4] Release Tagging: $TagName" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "[DryRun] Would commit changes, tag as $TagName, and push." -ForegroundColor Magenta
} else {
    git add -A
    $CommitMsg = "chore(release): spodeian-core v$Version"
    git commit -m $CommitMsg
    
    # Check if tag already exists
    $ExistingTag = git tag -l $TagName
    if ($ExistingTag) {
        Write-Host "Tag $TagName already exists. Updating tag..." -ForegroundColor Yellow
        git tag -d $TagName
    }
    
    git tag -a $TagName -m "Release spodeian-core v$Version"
    Write-Host "Created tag $TagName" -ForegroundColor Green

    if ($Push) {
        Write-Host "Pushing $CurrentBranch and tags to origin..." -ForegroundColor Cyan
        git push origin $CurrentBranch --tags
        Write-Host "Pushed successfully." -ForegroundColor Green
    } else {
        Write-Host "Push skipped (run with -Push to push automatically, or run: git push origin $CurrentBranch --tags)" -ForegroundColor Yellow
    }
}

Write-Host "`n=== Release v$Version Complete! ===" -ForegroundColor Green
