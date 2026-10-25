#!/usr/bin/env bash
# Spodeian Core Release Automation (Bash)
# Usage: ./scripts/release.sh -v "1.0.0" [-p] [-d]

set -euo pipefail

VERSION=""
PUSH=false
DRY_RUN=false

while getopts "v:pd" opt; do
  case ${opt} in
    v ) VERSION=$OPTARG;;
    p ) PUSH=true;;
    d ) DRY_RUN=true;;
    \? ) echo "Usage: ./scripts/release.sh -v <version> [-p] [-d]" && exit 1;;
  esac
done

if [ -z "$VERSION" ]; then
    echo "Error: Version (-v) is required. Example: ./scripts/release.sh -v 1.0.1"
    exit 1
fi

echo "=== Spodeian Core Release Automation: v$VERSION ==="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Active Git Branch: $CURRENT_BRANCH"

# Bump version in root Cargo.toml if necessary
if grep -q '\[workspace.package\]' Cargo.toml; then
    sed -i "s/^\(version = \)\"[^\"]*\"/\1\"$VERSION\"/" Cargo.toml
    echo "Updated Cargo.toml version to $VERSION"
fi

echo ""
echo "[1/4] Running workspace tests..."
cargo test --workspace

echo ""
echo "[2/4] Running workspace clippy check..."
cargo clippy --workspace -- -D warnings

echo ""
echo "[3/4] Checking semver contracts..."
if command -v cargo-semver-checks &> /dev/null; then
    cargo semver-checks check-release
else
    echo "cargo-semver-checks not installed in environment; skipping automated breaking change check."
    echo "(Install with: cargo install cargo-semver-checks --locked)"
fi

TAG_NAME="spodeian-core-v$VERSION"
echo ""
echo "[4/4] Release Tagging: $TAG_NAME"

if [ "$DRY_RUN" = true ]; then
    echo "[DryRun] Would commit changes, tag as $TAG_NAME, and push."
else
    git add -A
    git commit -m "chore(release): spodeian-core v$VERSION" || true

    if git tag -l "$TAG_NAME" | grep -q "$TAG_NAME"; then
        echo "Tag $TAG_NAME already exists. Updating tag..."
        git tag -d "$TAG_NAME"
    fi

    git tag -a "$TAG_NAME" -m "Release spodeian-core v$VERSION"
    echo "Created tag $TAG_NAME"

    if [ "$PUSH" = true ]; then
        echo "Pushing $CURRENT_BRANCH and tags to origin..."
        git push origin "$CURRENT_BRANCH" --tags
        echo "Pushed successfully."
    else
        echo "Push skipped (run with -p to push automatically, or run: git push origin $CURRENT_BRANCH --tags)"
    fi
fi

echo ""
echo "=== Release v$VERSION Complete! ==="
