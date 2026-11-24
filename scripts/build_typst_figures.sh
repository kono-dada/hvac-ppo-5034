#!/usr/bin/env bash
# Build the Typst figures: export the data, then compile every figure to PDF, SVG and PNG.
#
# Usage:  scripts/build_typst_figures.sh [figure-name ...]    (default: all figures)
# Output: docs/report/typst/build/<name>.{pdf,svg,png}
#
# Uses the Typst binary, package cache and fonts kept inside the repository (tools/),
# so nothing has to be installed system-wide. Set SKIP_EXPORT=1 to reuse the JSON data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TYPST="$ROOT/tools/typst/typst"
SRC="$ROOT/docs/report/typst"
OUT="$SRC/build"
export TYPST_PACKAGE_CACHE_PATH="$ROOT/tools/typst/packages"
export TYPST_FONT_PATHS="$ROOT/tools/fonts"

FIGS=(env-schematic env-step-response env-days env-controllers method-loop
      ablation-learning-curves ablation-final ablation-tradeoff ablation-update-size)
[ $# -gt 0 ] && FIGS=("$@")

if [ "${SKIP_EXPORT:-0}" != "1" ]; then
  (cd "$ROOT" && MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python scripts/export_figure_data.py)
fi

mkdir -p "$OUT"
for f in "${FIGS[@]}"; do
  for fmt in pdf svg png; do
    extra=()
    [ "$fmt" = png ] && extra=(--ppi 250)
    "$TYPST" compile --root "$SRC" --ignore-system-fonts --input "fig=$f" ${extra[@]+"${extra[@]}"} \
      "$SRC/figures/standalone.typ" "$OUT/$f.$fmt"
  done
  echo "built $f"
done
