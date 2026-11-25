// Layout of the report: a two-column A4 course report in the style of a LaTeX article
// (\documentclass[10pt,twocolumn]{article}).
//
// Typography: everything (title, headings, body, captions, tables) in New Computer Modern,
// maths in New Computer Modern Math; both ship with Typst, so no system fonts are needed.
// The figures use the same typeface (figures/style.typ) and are placed at natural size, so their
// 8pt text matches the tables.
//
// This file only defines how things look. Content lives in sections/, shared data and
// helpers in lib.typ, and the title-page information in report.typ.

// ---------------------------------------------------------------- palette
// Shared with the figures (figures/style.typ) for the verdict marks and table rules.
#let ink = rgb("#1d232b")
#let muted = rgb("#6b7380")
#let faint = rgb("#dfe3e8")
#let good = rgb("#3f9256")
#let warn = rgb("#c98a1b")
#let bad = rgb("#d1453b")
#let link-colour = rgb("#1f4e8c")

#let serif = "New Computer Modern"
#let mono = "DejaVu Sans Mono"

// Body text size; everything else is relative to it.
#let body-size = 10pt
// Lines are measured from the font's ascender to its descender rather than from the cap
// height to the baseline, so that sub- and superscripts and small inline maths fit inside
// the line and do not push individual lines apart. Baseline-to-baseline distance is then
// about 1em + leading (here 12.2pt, close to LaTeX's 12pt at 10pt).
#let leading = 0.3em

// ---------------------------------------------------------------- small helpers
/// A visible placeholder for information that still has to be filled in.
#let todo(body) = box(
  fill: rgb("#fff3cd"),
  stroke: (paint: rgb("#d39e00"), thickness: 0.6pt, dash: "dashed"),
  radius: 2pt,
  inset: (x: 4pt, y: 0pt),
  outset: (y: 2.5pt),
  text(fill: rgb("#7a5a00"), body),
)

/// Run-in paragraph heading, like LaTeX's \paragraph.
#let para(title) = strong(title) + h(0.6em)

// ---------------------------------------------------------------- the template
#let report(
  title: none,
  subtitle: none,
  course: none,
  authors: (),
  date: none,
  abstract: none,
  body,
) = {
  set document(title: title)
  set page(
    paper: "a4",
    margin: (x: 1.9cm, top: 2.0cm, bottom: 2.2cm),
    columns: 2,
    numbering: "1",
    number-align: center,
  )
  set columns(gutter: 0.7cm)
  set text(font: serif, size: body-size, fill: ink, lang: "en", top-edge: "ascender", bottom-edge: "descender")
  set par(justify: true, leading: leading, spacing: leading, first-line-indent: 1.5em)
  show raw: set text(font: mono, size: 0.82em)
  // The centred dot of New Computer Modern is set with wide side bearings; tighten the unit.
  show "K·h": box[K#h(-0.1em)·#h(-0.1em)h]

  // headings: numbered, bold, same family as the text
  set heading(numbering: "1.1")
  show heading: set par(justify: false, first-line-indent: 0pt)
  show heading: it => {
    // Absolute sizes, a step below LaTeX's defaults because the columns are narrow.
    let (size, above, below, number-width) = if it.level == 1 {
      (12pt, 17pt, 8pt, 1.5em)
    } else {
      (10.5pt, 13pt, 6pt, 2.2em)
    }
    set text(size: size, weight: "bold")
    block(above: above, below: below, sticky: true, {
      if it.numbering != none {
        // hanging indent: a wrapped title stays aligned with its first line
        grid(columns: (number-width, 1fr), counter(heading).display(it.numbering), it.body)
      } else {
        it.body
      }
    })
  }

  // maths
  set math.equation(numbering: "(1)", supplement: [Eq.])
  show math.equation.where(block: true): set block(above: 0.8em, below: 0.8em)

  // lists
  set list(indent: 0.6em, body-indent: 0.5em, spacing: leading)
  set enum(indent: 0.6em, body-indent: 0.5em, spacing: leading)
  show list: set block(above: 0.6em, below: 0.6em)
  show enum: set block(above: 0.6em, below: 0.6em)

  // figures and tables
  set figure(gap: 0.8em)
  set place(clearance: 1.4em)
  show figure.caption: it => {
    set text(size: 9pt)
    set par(justify: true, leading: 0.24em, first-line-indent: 0pt)
    set align(left)
    [*#it.supplement #context it.counter.display(it.numbering).* #it.body]
  }
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: table): set figure(gap: 0.7em)
  set table(stroke: none, inset: (x: 3pt, y: 2.6pt))
  // Size set on the figure, so that a table can still choose a smaller one for itself.
  show figure.where(kind: table): set text(size: 8pt)
  show table: set text(number-width: "tabular")
  show table: set par(justify: false, leading: 0.2em, first-line-indent: 0pt)

  // references and links
  show cite: set text(fill: link-colour)
  show ref: set text(fill: link-colour)
  show link: set text(fill: link-colour)
  set bibliography(style: "ieee", title: [References])
  show bibliography: set text(size: 8.5pt)
  show bibliography: set par(justify: false, leading: 0.22em, spacing: 0.6em, first-line-indent: 0pt)

  // ---- title block, across both columns (like \twocolumn[...] with \maketitle)
  place(top + center, float: true, scope: "parent", clearance: 1.8em, {
    set align(center)
    set par(justify: false, first-line-indent: 0pt, leading: 0.25em)
    v(0.3cm)
    text(size: 17pt, weight: "bold", title)
    if subtitle != none {
      v(0.8em, weak: true)
      text(size: 12pt, subtitle)
    }
    v(1.4em, weak: true)
    {
      set text(size: 11pt)
      authors.join(h(2em))
      if course != none { linebreak(); course }
      if date != none { linebreak(); date }
    }
    if abstract != none {
      v(1.6em, weak: true)
      block(width: 82%, {
        set text(size: 9.5pt)
        set par(justify: true, first-line-indent: 0pt, leading: 0.28em)
        set align(left)
        align(center, strong[Abstract])
        v(0.6em, weak: true)
        abstract
      })
    }
  })

  body
}
