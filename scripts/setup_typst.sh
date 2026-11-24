#!/usr/bin/env bash
# Download the Typst toolchain into tools/ (nothing is installed system-wide).
#   tools/typst/typst        Typst 0.15.1 binary (macOS arm64/x86_64 or Linux x86_64)
#   tools/typst/packages/    package cache (lilaq, cetz and their dependencies)
#   tools/docs/              offline documentation: Typst, lilaq, CeTZ, fletcher
# The fonts in tools/fonts/ (Fira Sans, Fira Math; SIL OFL) are kept in the repository.
#
# Usage:  scripts/setup_typst.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION=0.15.1
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) TARGET=aarch64-apple-darwin ;;
  Darwin-x86_64) TARGET=x86_64-apple-darwin ;;
  Linux-x86_64) TARGET=x86_64-unknown-linux-musl ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m)"; exit 1 ;;
esac

mkdir -p "$ROOT/tools/typst" "$ROOT/tools/docs"
if [ ! -x "$ROOT/tools/typst/typst" ]; then
  curl -fsSL "https://github.com/typst/typst/releases/download/v$VERSION/typst-$TARGET.tar.xz" \
    | tar xJ -C "$ROOT/tools/typst" --strip-components=1
fi
"$ROOT/tools/typst/typst" --version

# packages: compiling the figures once downloads everything they import into the local cache
export TYPST_PACKAGE_CACHE_PATH="$ROOT/tools/typst/packages"
printf '#import "@preview/lilaq:0.6.0"\n#import "@preview/cetz:0.5.3"\n#import "@preview/fletcher:0.6.0"\n' \
  > /tmp/typst-warmup.typ
"$ROOT/tools/typst/typst" compile /tmp/typst-warmup.typ /tmp/typst-warmup.pdf

# offline documentation
cd "$ROOT/tools/docs"
if [ ! -d typst ]; then
  mkdir -p typst && curl -fsSL "https://codeload.github.com/typst/typst/tar.gz/refs/tags/v$VERSION" \
    | tar xz -C typst --strip-components=1 "typst-$VERSION/docs"
fi
if [ ! -d lilaq-site ]; then  # lilaq's documentation website (markdown sources)
  mkdir -p lilaq-site && curl -fsSL https://codeload.github.com/lilaq-project/lilaq-project.github.io/tar.gz/HEAD \
    | tar xz -C lilaq-site --strip-components=1
fi
[ -f cetz-manual.pdf ] || curl -fsSL -o cetz-manual.pdf https://github.com/cetz-package/cetz/raw/refs/tags/v0.5.3/manual.pdf
if [ ! -d fletcher ]; then
  mkdir -p fletcher && curl -fsSL https://codeload.github.com/fletcher-package/fletcher/tar.gz/refs/tags/v0.6.0 \
    | tar xz -C fletcher --strip-components=1 fletcher-0.6.0/docs fletcher-0.6.0/README.md
fi
echo "done: $ROOT/tools"
