#!/usr/bin/env bash
# Build the report: export the numbers behind its tables, then compile the PDF.
#
# Usage:   scripts/build_report.sh
# Output:  docs/report/report.pdf                          the report
#          docs/report/typst/build/report-pages/N.png      one image per page, for a quick look
#
# Environment:
#   SKIP_EXPORT=1   reuse docs/report/typst/data/report.json (skips replaying the saved policies)
#   SKIP_PAGES=1    do not render the page images
#
# Uses the Typst binary and package cache in tools/ (see scripts/setup_typst.sh) and the fonts in
# tools/fonts; system fonts are ignored, so the result is the same on every machine.
# The data of the figures (docs/report/typst/figures/data/) is not rebuilt here; that is done by
# scripts/build_typst_figures.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TYPST="$ROOT/tools/typst/typst"
SRC="$ROOT/docs/report/typst"
PDF="$ROOT/docs/report/report.pdf"
PAGES="$SRC/build/report-pages"
export TYPST_PACKAGE_CACHE_PATH="$ROOT/tools/typst/packages"

[ -x "$TYPST" ] || { echo "Typst not found at $TYPST; run scripts/setup_typst.sh first" >&2; exit 1; }

if [ "${SKIP_EXPORT:-0}" != "1" ]; then
  (cd "$ROOT" && MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl" PYTHONPATH=. .venv/bin/python scripts/export_report_data.py >/dev/null)
  echo "exported docs/report/typst/data/report.json"
fi

compile() { "$TYPST" compile --root "$SRC" --ignore-system-fonts --font-path "$ROOT/tools/fonts" "$@"; }

compile "$SRC/report.typ" "$PDF"
echo "built ${PDF#"$ROOT"/}"

if [ "${SKIP_PAGES:-0}" != "1" ]; then
  mkdir -p "$PAGES"
  rm -f "${PAGES:?}"/*.png
  compile --ppi 110 "$SRC/report.typ" "$PAGES/{p}.png"
  echo "rendered $(ls "$PAGES" | wc -l | tr -d ' ') page image(s) in ${PAGES#"$ROOT"/}"
fi
