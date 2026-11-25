// Shared by every section: the numbers behind the tables, number formatting,
// and the helpers that place figures and tables.
//
// Sections start with  #import "../lib.typ": *

#import "template.typ": *

// ---------------------------------------------------------------- data
// Written by scripts/export_report_data.py; nothing in the report's tables is typed by hand.
#let data = json("data/report.json")
#let ab = data.ablation
#let base = ab.baseline          // the default configuration
#let groups = ab.groups          // one group per ablated hyperparameter
#let totals = ab.verdict_totals  // how the pre-registered predictions fared

/// One configuration by its run name: cfg("clip_0.5").cost
#let cfg(name) = {
  if name == "baseline" { return base }
  for g in groups {
    for r in g.rows {
      if r.name == name { return r }
    }
  }
  panic("unknown configuration " + name)
}

/// One reference controller by key: ctrl("oracle").cost
#let ctrl(key) = data.controllers.find(c => c.key == key)

// ---------------------------------------------------------------- numbers
/// Number with a fixed number of decimals and a typographic minus.
#let num(x, digits: 2, sign: false) = {
  let s = str(calc.round(calc.abs(x), digits: digits))
  if digits > 0 {
    let parts = s.split(".")
    let frac = if parts.len() > 1 { parts.at(1) } else { "" }
    s = parts.at(0) + "." + frac + "0" * (digits - frac.len())
  }
  let zero = float(s) == 0.0
  if x < 0 and not zero { "−" + s } else if sign and not zero { "+" + s } else { s }
}

/// A value of one configuration: val("clip_0.5", "cost") -> 5.11
#let val(name, key, digits: 2) = num(cfg(name).at(key), digits: digits)

/// Difference from the baseline with its 95% Welch interval: +0.37 [+0.04, +0.70]
#let interval(d, digits: 2) = [#num(d.d, digits: digits, sign: true) #h(0.15em) \[#num(d.lo, digits: digits, sign: true), #num(d.hi, digits: digits, sign: true)\]]
#let diff(name, key: "cost", digits: 2) = interval(cfg(name).diff.at(key), digits: digits)

/// Training steps in thousands: 204800 -> 205k
#let ksteps(n) = [#calc.round(n / 1000)k]

/// Value of an ablated setting as written in tables: 0.05, 1, 30
#let setting(v) = str(v)

// ---------------------------------------------------------------- verdict marks
/// Verdict on one prediction as a small symbol that does not rely on colour alone:
/// filled = held, half-filled = partly held, cross = failed, hollow = undecided.
#let mark(kind, size: 0.62em) = box(width: size, height: size, baseline: 0.02em, {
  if kind == "hit" {
    circle(radius: size / 2, fill: good, stroke: 0.5pt + good)
  } else if kind == "partial" {
    box(width: size, height: size, radius: size / 2, clip: true, stroke: 0.6pt + warn,
      place(left + top, rect(width: size / 2, height: size, fill: warn)))
  } else if kind == "miss" {
    place(line(start: (6%, 6%), end: (94%, 94%), stroke: (paint: bad, thickness: 1.1pt, cap: "round")))
    place(line(start: (6%, 94%), end: (94%, 6%), stroke: (paint: bad, thickness: 1.1pt, cap: "round")))
  } else {
    circle(radius: size / 2, fill: none, stroke: 0.6pt + muted)
  }
})
#let verdict-names = (hit: [held], partial: [partly held], miss: [failed], undecided: [undecided])
#let verdict-order = ("hit", "partial", "miss", "undecided")

/// All verdicts of one configuration in a row: ●4 ×1
#let scorecard(counts) = verdict-order.filter(k => counts.at(k) > 0).map(k => box[#mark(k)#h(0.18em)#counts.at(k)]).join(h(0.5em))

/// The legend of the marks, for captions.
#let mark-legend = verdict-order.map(k => box[#mark(k)#h(0.25em)#verdict-names.at(k)]).join([, ])

// ---------------------------------------------------------------- figures and tables
/// Wraps a figure module's `render` for use in a figure: the body receives the available width.
/// `shrink` < 1 renders the figure larger and scales it down, which makes it less tall.
#let from-module(render, shrink: 1.0) = width => {
  if shrink == 1.0 {
    render(width: width, title: false)
  } else {
    scale(shrink * 100%, origin: top + left, reflow: true, render(width: width / shrink, title: false))
  }
}

/// A figure spanning both columns. `body` is a function of the available width.
#let wide-figure(body, caption: none, label: none, placement: top) = [
  #figure(
    placement: placement,
    scope: "parent",
    caption: caption,
    layout(size => {
      // Text settings of the report must not leak into the figures.
      set par(justify: false, first-line-indent: 0pt, leading: 0.65em)
      set text(top-edge: "cap-height", bottom-edge: "baseline")
      body(size.width)
    }),
  )
  #label
]

/// A figure inside one column. `body` is a function of the available width.
#let column-figure(body, caption: none, label: none, placement: auto) = [
  #figure(
    placement: placement,
    caption: caption,
    layout(size => {
      set par(justify: false, first-line-indent: 0pt, leading: 0.65em)
      set text(top-edge: "cap-height", bottom-edge: "baseline")
      body(size.width)
    }),
  )
  #label
]

/// A table inside one column.
#let column-table(body, caption: none, label: none, placement: auto) = [
  #figure(placement: placement, caption: caption, body)
  #label
]

/// A table spanning both columns.
#let wide-table(body, caption: none, label: none, placement: top) = [
  #figure(placement: placement, scope: "parent", caption: caption, body)
  #label
]

// Horizontal rules of tables.
#let rule-heavy = table.hline(stroke: 0.8pt + ink)
#let rule-light = table.hline(stroke: 0.4pt + muted)
#let rule-faint = table.hline(stroke: 0.4pt + faint)
