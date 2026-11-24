// Results at the end of training: every setting on one shared scale per metric, against its prediction.
#import "style.typ": *

#let d = json("data/ablation.json")
#let order = ("epochs", "clip", "lambda")
#let n-seeds = d.configs.baseline.seeds.len()
#let metrics = (
  (key: "cost", title: [Cost], unit: [energy + #fmt(d.weight, digits: 0) × discomfort], lim: (4.3, 6.75),
    ticks: (4.5, 5.0, 5.5, 6.0, 6.5)),
  (key: "energy_kwh", title: [Energy], unit: [kWh/day], lim: (4.1, 5.7), ticks: (4.5, 5.0, 5.5)),
  (key: "discomfort_kh", title: [Discomfort], unit: [K·h/day], lim: (-0.015, 0.43), ticks: (0, 0.1, 0.2, 0.3, 0.4)),
)
// fixed vertical offsets of the seeds within a row
#let jitter = (-0.2, 0.13, -0.07, 0.2, -0.13, 0.07, 0.0, -0.25, 0.25, -0.03, 0.16, -0.16)

// one header row per group, then one row per value; rows run downwards (y = 0, -1, -2, ...)
#let rows = {
  let out = ()
  let y = 0
  for g in order {
    out.push((kind: "header", group: g, y: -y))
    y += 1
    let grp = d.groups.at(g)
    let colors = group-colors(grp)
    for (i, name) in grp.names.enumerate() {
      out.push((kind: "row", group: g, name: name, value: grp.values.at(i), color: colors.at(i), y: -y))
      y += 1
    }
    y += 0.5
  }
  out
}
#let y-bottom = rows.last().y - 0.7

#let value-label(r) = if r.name == "baseline" {
  [#text(fill: ink, weight: "bold")[#r.value] #h(0.25em) #text(size: 0.92em)[baseline]]
} else [#text(fill: ink-2)[#r.value]]

#let strip(m, letter, labels: false) = {
  let (lo, hi) = m.lim
  let base = d.configs.baseline.final.at(m.key)
  let thermo = d.refs.thermostat.at(m.key)
  let objs = ()
  // the baseline: 95% CI as a vertical band through all rows, its mean as a thin line
  objs.push(lq.rect(base.lo, 0%, width: base.hi - base.lo, height: 100%, fill: ink.transparentize(92%), stroke: none,
    z-index: 0))
  objs.push(lq.vlines(base.mean, stroke: (paint: ink.transparentize(55%), thickness: 0.6pt), z-index: 1))
  // the tuned thermostat
  objs.push(lq.vlines(thermo, stroke: (paint: thermostat-c, thickness: 1pt, dash: (3pt, 2pt)), z-index: 1))
  objs.push(lq.place(thermo, rows.first().y, align: left + horizon, pad(left: 3pt, text(fill: thermostat-c,
    size: 0.92em, weight: "medium")[thermostat])))
  for r in rows {
    if r.kind == "header" {
      if r.y < 0 { objs.push(lq.hlines(r.y + 0.75, stroke: 0.5pt + faint, z-index: 0)) }
      continue
    }
    let f = d.configs.at(r.name).final.at(m.key)
    if r.name == "baseline" {
      objs.push(lq.rect(0%, r.y - 0.5, width: 100%, height: 1, fill: soft, stroke: none, z-index: 0))
    }
    if m.key == "cost" and r.name in d.predicted_cost {
      let (p0, p1) = d.predicted_cost.at(r.name)
      objs.push(lq.rect(p0, r.y - 0.36, width: p1 - p0, height: 0.72, fill: prediction-soft, radius: 1.5pt,
        stroke: 0.6pt + prediction-c.transparentize(35%), z-index: 1))
    }
    if f.mean > hi {
      // off the scale: write the result (and, for the cost, the prediction) at the edge instead
      let x-end = if m.key == "cost" and r.name in d.predicted_cost { d.predicted_cost.at(r.name).at(0) } else { hi }
      objs.push(lq.place(x-end, r.y, align: right + horizon, pad(right: 4pt, box(text(fill: r.color, size: 0.94em)[
        *#fmt(f.mean, digits: 2)* [#fmt(f.lo, digits: 2), #fmt(f.hi, digits: 2)] #sym.arrow.r]))))
    } else {
      let seeds = f.values.enumerate()
      let inside = seeds.filter(((i, v)) => v <= hi)
      let outside = seeds.filter(((i, v)) => v > hi)
      objs.push(lq.scatter(inside.map(p => p.at(1)), inside.map(p => r.y + jitter.at(p.at(0))), size: 2.6pt,
        color: r.color.transparentize(50%), stroke: none, z-index: 2))
      if outside.len() > 0 {
        objs.push(lq.scatter(outside.map(_ => hi - (hi - lo) * 0.012), outside.map(p => r.y + jitter.at(p.at(0))),
          mark: ">", size: 3.6pt, color: r.color.transparentize(25%), stroke: none, z-index: 2))
      }
      objs.push(lq.plot((f.lo, f.hi), (r.y, r.y), mark: none, z-index: 3,
        stroke: (paint: r.color, thickness: 2.6pt, cap: "round")))
      objs.push(lq.scatter((f.mean,), (r.y,), size: 4.6pt, color: r.color, stroke: 0.9pt + white, z-index: 4))
    }
    if m.key == "cost" and r.name in d.verdicts {
      objs.push(lq.place(hi, r.y, align: left + horizon, pad(left: 5pt, verdict-mark(d.verdicts.at(r.name)))))
    }
  }
  show: lq.cond-set(lq.grid.with(kind: "x"), stroke: 0.5pt + hairline)
  show: lq.cond-set(lq.grid.with(kind: "y"), stroke: none)
  lq.diagram(
    width: 100%,
    height: 7.6cm,
    title: panel(letter, m.title, unit: m.unit),
    xlim: m.lim,
    ylim: (y-bottom, 0.6),
    xaxis: (ticks: m.ticks.map(t => (t, fmt(t, digits: 1)))),
    yaxis: (ticks: rows.map(r => (r.y, if not labels { [] } else if r.kind == "header" {
      text(fill: ink, weight: "bold", group-title.at(r.group))
    } else { value-label(r) }))),
    ..objs,
  )
}

#let render(width: text-width, title: true) = {
  show: theme
  if title {
    heading-block(width: width, [Where each setting ends up after training, and how the predictions fared],
      sub: [#n-seeds seeds per setting, evaluated on the 100 test days. Settings below the baseline value in blues,
        above it in oranges and reds. The verdict next to each cost compares its 95% interval with the range predicted
        in advance.])
  }
  let ci-swatch(c) = box(width: 1.6em, height: 0.7em, baseline: 0.05em, {
    place(horizon, line(start: (15%, 0%), end: (85%, 0%), stroke: (paint: c, thickness: 2.6pt, cap: "round")))
    place(center + horizon, circle(radius: 2.3pt, fill: c, stroke: 0.9pt + white))
  })
  block(width: width, below: 0.8em, {
    legend-row(gap: 1.0em,
      (ci-swatch(ink), [mean and 95% CI]),
      (box(width: 0.8em, align(center, dot(ink.transparentize(50%), size: 2.8pt, stroke: none))), [one seed]),
      (swatch(kind: "area", color: prediction-c.transparentize(35%), fill: prediction-soft, width: 1.4em),
        [predicted cost range]),
      (swatch(kind: "area", color: ink.transparentize(92%), width: 1em), [baseline 95% CI]),
      (swatch(color: thermostat-c, thickness: 1pt, dash: (3pt, 2pt)), [tuned thermostat]),
    )
    linebreak()
    box[#text(size: 0.96em)[Cost prediction (right of panel a): #h(0.3em) #verdict-mark("hit") held #h(0.6em)
      #verdict-mark("partial") partly held, the intervals overlap #h(0.6em) #verdict-mark("miss") missed]]
  })
  show: lq.layout
  block(width: width, std.grid(
    columns: (1.62fr, 1fr, 1fr),
    column-gutter: (0.85cm, 0.55cm),
    strip(metrics.at(0), "a", labels: true),
    strip(metrics.at(1), "b"),
    strip(metrics.at(2), "c"),
  ))
}
