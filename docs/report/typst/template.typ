// Layout of the report: a compact two-column A4 paper.
//
// Typography: body text in Libertinus Serif (ships with Typst); headings, captions and
// tables in Fira Sans, the typeface of the figures (figures/style.typ), so that text and
// figures read as one piece. No system fonts are needed (Fira lives in tools/fonts).
//
// This file only defines how things look. Content lives in sections/, shared data and
// helpers in lib.typ, and the title-page information in report.typ.

// ---------------------------------------------------------------- palette (matches figures/style.typ)
#let ink = rgb("#1d232b")
#let muted = rgb("#6b7380")
#let faint = rgb("#dfe3e8")
#let soft = rgb("#f3f5f7")
#let accent = rgb("#24558f")
#let good = rgb("#3f9256")
#let warn = rgb("#c98a1b")
#let bad = rgb("#d1453b")

#let serif = "Libertinus Serif"
#let sans = "Fira Sans"
#let mono = "DejaVu Sans Mono"

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

/// Run-in paragraph heading.
#let para(title) = text(font: sans, size: 0.9em, weight: "semibold", fill: ink, title) + h(0.55em)

/// Text set like a table or caption (sans, small); for notes under tables.
#let small(body) = {
  set text(font: sans, size: 7.2pt, fill: muted)
  set par(justify: false, leading: 0.48em, first-line-indent: 0pt)
  body
}

// ---------------------------------------------------------------- the template
#let report(
  title: none,
  subtitle: none,
  kicker: none,
  authors: (),
  date: none,
  abstract: none,
  keywords: (),
  running-title: none,
  body,
) = {
  set document(title: title)
  set page(
    paper: "a4",
    margin: (x: 1.8cm, top: 1.8cm, bottom: 2.0cm),
    columns: 2,
    footer: context {
      set text(font: sans, size: 7.2pt, fill: muted)
      grid(
        columns: (1fr, auto),
        running-title,
        [#counter(page).display() / #counter(page).final().first()],
      )
    },
  )
  set columns(gutter: 0.6cm)
  set text(font: serif, size: 9.3pt, fill: ink, lang: "en", number-type: "lining")
  set par(justify: true, leading: 0.56em, spacing: 0.56em, first-line-indent: 1em)
  show raw: set text(font: mono, size: 0.84em)

  // headings
  set heading(numbering: "1.1")
  show heading: set par(first-line-indent: 0pt, justify: false)
  show heading.where(level: 1): it => block(above: 1.3em, below: 0.75em, sticky: true, {
    set text(font: sans, size: 10.6pt, weight: "semibold", fill: ink)
    if it.numbering != none {
      text(fill: accent, counter(heading).display(it.numbering))
      h(0.65em)
    }
    it.body
  })
  show heading.where(level: 2): it => block(above: 1.05em, below: 0.65em, sticky: true, {
    set text(font: sans, size: 9.1pt, weight: "semibold", fill: ink)
    if it.numbering != none {
      text(fill: accent, counter(heading).display(it.numbering))
      h(0.6em)
    }
    it.body
  })

  // maths
  set math.equation(numbering: "(1)", supplement: [Eq.])
  show math.equation.where(block: true): set block(above: 0.85em, below: 0.85em)

  // lists
  set list(indent: 0.3em, body-indent: 0.5em, spacing: 0.56em, marker: text(fill: accent, sym.bullet))
  set enum(indent: 0.3em, body-indent: 0.5em, spacing: 0.56em)

  // figures and tables
  set figure(gap: 0.7em)
  set place(clearance: 1.25em)
  show figure.caption: it => {
    set text(font: sans, size: 7.7pt, fill: ink)
    set par(justify: false, leading: 0.5em, first-line-indent: 0pt)
    set align(left)
    text(weight: "semibold", fill: accent)[#it.supplement #context it.counter.display(it.numbering)]
    h(0.55em)
    it.body
  }
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: table): set figure(gap: 0.55em)
  set table(stroke: none, inset: (x: 3.2pt, y: 2.7pt))
  show table: set text(font: sans, size: 7.5pt, number-type: "lining", number-width: "tabular")
  show table: set par(justify: false, leading: 0.45em, first-line-indent: 0pt)

  // references and links
  show cite: set text(fill: accent)
  show ref: set text(fill: accent)
  show link: set text(fill: accent)
  set bibliography(style: "ieee", title: [References])
  show bibliography: set text(size: 7.6pt)
  show bibliography: set par(justify: false, leading: 0.48em, spacing: 0.5em, first-line-indent: 0pt)

  // ---- title block, across both columns
  place(top + center, float: true, scope: "parent", clearance: 1.6em, {
    set par(first-line-indent: 0pt, justify: false)
    set align(left)
    if kicker != none {
      text(font: sans, size: 8pt, weight: "medium", fill: accent, tracking: 0.07em, upper(kicker))
      v(0.7em, weak: true)
    }
    text(font: sans, size: 19pt, weight: "bold", fill: ink, title)
    if subtitle != none {
      v(0.6em, weak: true)
      text(font: sans, size: 12pt, weight: "regular", fill: muted, subtitle)
    }
    v(1.1em, weak: true)
    {
      set text(font: sans, size: 9pt)
      authors.join(h(1.6em))
      if date != none {
        h(1fr)
        text(fill: muted, date)
      }
    }
    v(1.1em, weak: true)
    block(
      width: 100%,
      fill: soft,
      stroke: (left: 2.2pt + accent),
      inset: (left: 11pt, right: 11pt, y: 9pt),
      {
        set par(justify: true, leading: 0.56em)
        text(font: sans, size: 8.4pt, weight: "semibold", fill: accent)[Abstract]
        h(0.7em)
        abstract
        if keywords.len() > 0 {
          v(0.7em, weak: true)
          set text(font: sans, size: 7.7pt, fill: muted)
          [#text(weight: "semibold")[Keywords] #h(0.5em) #keywords.join([ · ])]
        }
      },
    )
  })

  body
}
