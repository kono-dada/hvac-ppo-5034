// Shared look of every report figure.
//
// Visual language
//   - The typeface of the report text: New Computer Modern, maths in New Computer Modern Math
//     (both ship with Typst). The family has only regular and bold, so emphasis is bold.
//   - One text size: figures are placed at their natural size (no scaling), so 8pt in a
//     figure is 8pt on the page, the size of the report's tables (captions are 9pt).
//   - Light chrome: no frame and no y spine, a thin x baseline, faint horizontal guides only.
//     Units go into the panel titles instead of rotated axis labels.
//   - Label lines and points directly where possible; legends only where that would crowd the plot.
//
// Colour conventions (the report text relies on them):
//   thermostat = orange, oracle = near-black (ink), ablation baseline = ink;
//   heating = red, cooling = blue, comfort band = green;
//   walls & floor (thermal mass) = brown, outdoors = grey-blue, sun = amber;
//   ablation: values below the baseline in blues, above it in oranges/reds,
//   darker the further from the baseline; predictions = violet;
//   verdicts: held = green, partly = amber, missed = red (as in the report template).

#import "@preview/lilaq:0.6.0" as lq

// ---------------------------------------------------------------- sizes
#let text-width = 16cm           // default figure width (the report passes its own)
#let base-size = 8pt
#let font = "New Computer Modern"
#let math-font = "New Computer Modern Math"

// ---------------------------------------------------------------- palette
// neutrals (ink, muted, faint, soft and accent are shared with the report template)
#let ink = rgb("#1d232b")
#let ink-2 = rgb("#3d4550")
#let muted = rgb("#6b7380")
#let subtle = rgb("#a3abb5")
#let faint = rgb("#dfe3e8")
#let hairline = rgb("#eaedf0")
#let soft = rgb("#f3f5f7")
#let accent = rgb("#24558f")

// controllers
#let thermostat-c = rgb("#e8832a")
#let oracle-c = ink
#let baseline-c = ink
#let off-c = rgb("#9ea6b0")

// physics
#let heat = rgb("#d1453b")
#let heat-soft = rgb("#fbe9e6")
#let cool = rgb("#2f6db5")
#let cool-soft = rgb("#e6eef8")
#let comfort = rgb("#3f9256")
#let comfort-soft = rgb("#e1f0e4")
#let air-c = ink
#let mass-c = rgb("#9a6a3f")
#let mass-soft = rgb("#f1e6da")
#let outdoor-c = rgb("#7f9dbf")
#let outdoor-dark = rgb("#4f6f93")
#let sun-c = rgb("#e9a227")
#let sun-dark = rgb("#b97d0c")
#let internal-c = rgb("#7b8794")

// predictions and verdicts
#let prediction-c = rgb("#7d68c6")
#let prediction-soft = rgb("#ece8f8")
#let good = rgb("#3f9256")
#let warn = rgb("#e0a21b")
#let bad = rgb("#d1453b")

// diverging ramps for the ablation, index 0 = nearest the baseline value
#let below-ramp = (rgb("#69a3d6"), rgb("#22609f"), rgb("#163d69"))
#let above-ramp = (rgb("#f0a044"), rgb("#d0502b"), rgb("#8d2413"))

/// Colour of one ablation value from its side (-1, 0, +1) and rank (0 = nearest).
#let ablation-color(side, rank) = if side == 0 { baseline-c } else if side < 0 {
  below-ramp.at(rank)
} else { above-ramp.at(rank) }

/// Colours of all values of one ablation group (from data/ablation.json).
#let group-colors(g) = g.side.zip(g.rank).map(((s, k)) => ablation-color(s, k))

/// Names of the ablation groups as printed.
#let group-title = (epochs: [n_epochs], clip: [clip range $epsilon$], lambda: [GAE $lambda$])

// ---------------------------------------------------------------- numbers
/// Number with a fixed number of decimals and a typographic minus.
#let fmt(x, digits: 1, sign: false) = {
  let s = str(calc.round(calc.abs(x), digits: digits))
  if digits > 0 {
    let parts = s.split(".")
    let frac = if parts.len() > 1 { parts.at(1) } else { "" }
    s = parts.at(0) + "." + frac + "0" * (digits - frac.len())
  }
  let zero = float(s) == 0.0
  if x < 0 and not zero { "−" + s } else if sign and not zero { "+" + s } else { s }
}

/// Spread label positions (same units as `ys`) so neighbours are at least `gap` apart,
/// keeping them inside `[lo, hi]` when given. Returns the positions in the input order.
#let dodge(ys, gap, lo: none, hi: none) = {
  let n = ys.len()
  if n == 0 { return () }
  let order = range(n).sorted(key: i => ys.at(i))
  let pos = order.map(i => ys.at(i))
  if lo != none and pos.first() < lo { pos.at(0) = lo }
  for k in range(1, n) {
    if pos.at(k) - pos.at(k - 1) < gap { pos.at(k) = pos.at(k - 1) + gap }
  }
  if hi != none and pos.last() > hi {
    pos.at(n - 1) = hi
    for k in range(n - 2, -1, step: -1) {
      if pos.at(k + 1) - pos.at(k) < gap { pos.at(k) = pos.at(k + 1) - gap }
    }
  }
  let out = ys
  for (k, i) in order.enumerate() { out.at(i) = pos.at(k) }
  out
}

// ---------------------------------------------------------------- text helpers
#let note(body) = text(fill: muted, body)

/// Panel heading: a letter in the accent colour, a semibold title and an optional grey unit.
#let panel(letter, body, unit: none) = {
  if letter != none {
    text(weight: "bold", fill: accent, letter)
    h(0.5em)
  }
  text(weight: "bold", body)
  if unit != none {
    h(0.4em)
    text(fill: muted, unit)
  }
}

/// A small rounded label with a background, for direct labelling inside plots.
#let tag(body, fill: white, color: ink, weight: "regular", inset: (x: 2pt, y: 1.2pt), radius: 1.5pt) = box(
  fill: fill,
  radius: radius,
  inset: inset,
  text(fill: color, weight: weight, body),
)

/// A legend swatch drawn by hand (for legends placed outside the diagrams).
#let swatch(kind: "line", color: ink, dash: none, thickness: 1.4pt, mark: none, width: 1.5em, fill: none) = box(
  width: width,
  height: 0.7em,
  baseline: 0.05em,
  {
    if kind == "line" {
      place(horizon, line(length: 100%, stroke: (paint: color, thickness: thickness, dash: dash, cap: "round")))
    } else if kind == "area" {
      place(horizon, rect(width: 100%, height: 0.8em, fill: if fill == none { color } else { fill }, radius: 1pt,
        stroke: if fill == none { none } else { 0.5pt + color }))
    }
    if mark != none { place(center + horizon, mark) }
  },
)

#let dot(color, size: 4.5pt, stroke: 0.6pt + white) = box(baseline: 0.5pt, circle(radius: size / 2, fill: color,
  stroke: stroke))

/// A row of (swatch, label) legend entries laid out inline.
#let legend-row(..entries, gap: 1.3em) = {
  entries.pos().map(((sw, label)) => box[#sw #h(0.3em) #label]).join(h(gap))
}

/// Verdict of a pre-registered prediction as a symbol that does not rely on colour alone
/// (same shapes as the report template): filled = held, half = partly, cross = missed.
#let verdict-mark(kind, size: 0.62em) = box(width: size, height: size, baseline: 0.02em, {
  if kind == "hit" {
    circle(radius: size / 2, fill: good, stroke: 0.5pt + good)
  } else if kind == "partial" {
    box(width: size, height: size, radius: size / 2, clip: true, stroke: 0.6pt + warn.darken(10%),
      place(left + top, rect(width: size / 2, height: size, fill: warn.darken(10%))))
  } else if kind == "miss" {
    place(line(start: (6%, 6%), end: (94%, 94%), stroke: (paint: bad, thickness: 1.1pt, cap: "round")))
    place(line(start: (6%, 94%), end: (94%, 6%), stroke: (paint: bad, thickness: 1.1pt, cap: "round")))
  } else {
    circle(radius: size / 2, fill: none, stroke: 0.6pt + muted)
  }
})
#let verdict-word = (hit: [held], partial: [partly held], miss: [missed], undecided: [undecided])

// ---------------------------------------------------------------- the theme
#let theme(body) = {
  set text(font: font, size: base-size, fill: ink, number-width: "tabular",
    top-edge: "cap-height", bottom-edge: "baseline")
  show math.equation: set text(font: math-font)
  set par(leading: 0.5em, spacing: 0.6em, justify: false)
  set align(left)

  // y axis: no spine and no tick marks (a transparent axis stroke hides both; tick settings are global)
  show: lq.set-diagram(
    xaxis: (subticks: none, mirror: false),
    yaxis: (subticks: none, mirror: false, stroke: (paint: white.transparentize(100%), thickness: 0.6pt)),
    margin: 0%,
  )
  // horizontal guides only
  show: lq.set-grid(stroke: 0.5pt + hairline)
  show: lq.cond-set(lq.grid.with(kind: "x"), stroke: none)
  show: lq.set-spine(stroke: (paint: subtle, thickness: 0.6pt, cap: "square"))
  show: lq.set-tick(inset: 0pt, outset: 2.5pt, pad: 0.4em)
  show lq.selector(lq.tick-label): set text(fill: muted)
  show lq.selector(lq.label): set text(fill: ink-2)
  show: lq.set-label(pad: 0.5em)
  show: lq.set-title(pad: 0.75em)
  show lq.selector(lq.title): set align(left)
  show: lq.set-legend(stroke: none, fill: white.transparentize(8%), pad: 0.3em, inset: 0.3em, radius: 1.5pt)
  show lq.selector(lq.legend): set grid(row-gutter: 0.1em, column-gutter: 0.6em)
  body
}

/// Figure heading (standalone builds only): a bold title and an optional grey subtitle.
#let heading-block(title, sub: none, width: text-width) = block(width: width, below: 1.0em, {
  text(size: 1.28em, weight: "bold", title)
  if sub != none {
    v(0.45em, weak: true)
    block(text(fill: muted, size: 1em, sub))
  }
})
