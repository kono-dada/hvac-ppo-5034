# Local Typst toolchain

Everything the Typst figures need lives in this folder, so nothing has to be installed system-wide.

| Path | Contents | In git? |
| --- | --- | --- |
| `typst/typst` | Typst 0.15.1 binary | no, run `scripts/setup_typst.sh` |
| `typst/packages/` | package cache: lilaq 0.6.0 (plots), CeTZ 0.5.3 (diagrams), fletcher 0.6.0, and their dependencies | no, same script |
| `docs/` | offline docs: `typst/docs` (language reference), `lilaq-site/docs` (lilaq tutorials, examples, reference), `cetz-manual.pdf`, `fletcher/docs` | no, same script |
| `fonts/` | Fira Sans and Fira Math (SIL Open Font License, see `OFL-*.txt`) | yes |

Build all figures (exports the data from the experiment runs, then compiles every figure to PDF, SVG and PNG):

```sh
scripts/setup_typst.sh          # once
scripts/build_typst_figures.sh  # output: docs/report/typst/build/
```

The scripts set `TYPST_PACKAGE_CACHE_PATH=tools/typst/packages` and `TYPST_FONT_PATHS=tools/fonts`.
When you call `tools/typst/typst` by hand, set the same two variables (or pass `--font-path tools/fonts`).
For the lilaq API, also see the package source in `typst/packages/preview/lilaq/0.6.0/src/`, which has doc comments on every parameter.
