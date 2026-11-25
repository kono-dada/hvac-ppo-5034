// Entry point of the report.
//
//   scripts/build_report.sh            builds docs/report/report.pdf
//
// Files
//   report.typ      this file: title-page information and the order of the sections
//   template.typ    page layout and typography
//   lib.typ         data (data/report.json), number formatting, figure and table helpers
//   sections/       one file per section
//   figures/        figure modules (each exports `render`), maintained separately
//   refs.bib        references

#import "template.typ": report

// ---------------------------------------------------------------- title page: edit here
#let course = [AIAA 5034]
#let authors = (
  [Fangda Chen (50040218)],
  [Liang Zhang (50040696)],
)
#let date = [November 2026]

#show: report.with(
  title: [Building HVAC Control with PPO],
  subtitle: [Effect of Clip Range, GAE Lambda, and Number of Epochs],
  course: [#course Final Project Report],
  authors: authors,
  date: date,
  abstract: include "sections/00-abstract.typ",
)

// ---------------------------------------------------------------- sections
#include "sections/01-introduction.typ"
#include "sections/02-environment.typ"
#include "sections/03-method.typ"
#include "sections/04-predictions.typ"
#include "sections/05-results.typ"
#include "sections/06-conclusion.typ"

#bibliography("refs.bib")
