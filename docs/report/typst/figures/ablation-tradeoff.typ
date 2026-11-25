// Energy versus discomfort: every configuration's end point on top of the baseline's own training path.
#import "style.typ": *

#let d = json("data/ablation.json")
#let w = d.weight
#let n-seeds = d.configs.baseline.seeds.len()
#let shape = (epochs: "s", clip: "^", lambda: "o")
#let label-of(g, v) = if g == "epochs" { if v == 1 [1 epoch] else [#v epochs] } else if g == "clip" [$epsilon$ #v] else [$lambda$ #v]

// zoomed panel: where each label sits relative to its point (dx, dy, alignment of the label)
#let offsets = (
  "epochs_3": (0pt, -6pt, center + bottom),
  "epochs_30": (-5pt, -3pt, right + bottom),
  "clip_0.05": (6pt, 0pt, left + horizon),
  "clip_0.1": (0pt, 6pt, center + top),
  "clip_0.3": (-6pt, 0pt, right + horizon),
  "clip_0.5": (6pt, 0pt, left + horizon),
  "lambda_0.8": (0pt, -6pt, center + bottom),
  "lambda_0.9": (6pt, 0pt, left + horizon),
  "lambda_0.99": (0pt, 6pt, center + top),
  "lambda_1": (4pt, -7pt, left + bottom),
)

/// A filled marker with a white outline.
#let point(x, y, m, color, size: 7pt, z: 5) = (
  lq.plot((x,), (y,), mark: m, mark-size: size + 2.4pt, color: white, stroke: none, z-index: z),
  lq.plot((x,), (y,), mark: m, mark-size: size, color: color, stroke: none, z-index: z + 0.1),
)
/// A label on a translucent white background, offset from a data point.
#let at(x, y, dx, dy, al, body) = lq.place(x, y, align: al,
  move(dx: dx, dy: dy, box(fill: white.transparentize(20%), radius: 1pt, inset: (x: 1.2pt, y: 0.8pt), body)))

/// Inline glyphs for the legend.
#let glyph(kind, color: muted) = box(width: 0.8em, height: 0.7em, baseline: 0.02em, align(center + horizon, {
  if kind == "s" { rect(width: 5.4pt, height: 5.4pt, fill: color) }
  else if kind == "^" { polygon.regular(vertices: 3, size: 7.4pt, fill: color) }
  else { circle(radius: 3pt, fill: color) }
}))

#let render(width: text-width, title: true) = {
  show: theme
  let base = d.configs.baseline
  let bf = base.final
  let keep = base.t.enumerate().filter(((i, t)) => t >= 16384).map(((i, _)) => i)
  let pe = keep.map(i => base.curves.energy_kwh.mean.at(i))
  let pd = keep.map(i => base.curves.discomfort_kh.mean.at(i))
  let refs = d.refs
  let (x0, x1, y0, y1) = (4.0, 5.45, -0.015, 0.4)
  let path-stroke = (paint: baseline-c, thickness: 1.5pt, join: "round")
  let iso-stroke = (paint: muted, thickness: 0.7pt, dash: (0.5pt, 2.2pt), cap: "round")
  let iso-costs = (
    (refs.thermostat.cost, [#fmt(refs.thermostat.cost, digits: 2) thermostat]),
    (bf.cost.mean, [#fmt(bf.cost.mean, digits: 2) baseline]),
    (5.0, [5.0]),
    (5.5, [5.5]),
  )

  // end points of every setting (and their 95% intervals when zoomed)
  let endpoints(zoom) = {
    let out = ()
    for (gname, g) in d.groups {
      let colors = group-colors(g)
      for (i, name) in g.names.enumerate() {
        if name == "baseline" { continue }
        let f = d.configs.at(name).final
        let (e, q) = (f.energy_kwh, f.discomfort_kh)
        let c = colors.at(i)
        if zoom and name not in offsets { continue }  // outside the zoomed range (1 epoch)
        if zoom {
          out.push(lq.line((e.lo, q.mean), (e.hi, q.mean), stroke: 1.2pt + c.transparentize(60%), z-index: 3))
          out.push(lq.line((e.mean, q.lo), (e.mean, q.hi), stroke: 1.2pt + c.transparentize(60%), z-index: 3))
          let (dx, dy, al) = offsets.at(name)
          out.push(at(e.mean, q.mean, dx, dy, al, text(fill: c.darken(8%), weight: "bold",
            label-of(gname, g.values.at(i)))))
        }
        out += point(e.mean, q.mean, shape.at(gname), c, size: if zoom { 7pt } else { 5pt })
      }
    }
    out
  }

  let overview = {
    let step-label(k, dx, dy, al) = {
      let i = base.t.position(t => t == k)
      at(base.curves.energy_kwh.mean.at(i), base.curves.discomfort_kh.mean.at(i), dx, dy, al,
        note[#calc.round(k / 1000)k])
    }
    let e1 = d.configs.epochs_1.final
    lq.diagram(
      width: 100%,
      height: 5.2cm,
      title: panel("a", [Whole training path]),
      ylabel: [Discomfort (K·h/day)],
      xlim: (3.1, 8.7),
      ylim: (-0.03, 0.47),
      xlabel: [Energy (kWh/day)],
      xaxis: (ticks: range(4, 9)),
      yaxis: (ticks: (0, 0.1, 0.2, 0.3, 0.4).map(t => (t, fmt(t, digits: 1)))),
      lq.rect(x0, y0, width: x1 - x0, height: y1 - y0, fill: soft, stroke: none, z-index: 0),
      lq.place(x0, y1, align: left + bottom, pad(bottom: 1pt, tag(fill: white.transparentize(15%), color: muted, inset: (x: 1pt, y: 1pt))[zoomed in b])),
      lq.plot(pe, pd, color: baseline-c, stroke: path-stroke, mark: "o", mark-size: 2.4pt, z-index: 2),
      step-label(16384, -5pt, 0pt, right + horizon),
      step-label(32768, 0pt, -5pt, center + bottom),
      step-label(65536, 2pt, -5pt, left + bottom),
      ..endpoints(false),
      at(e1.energy_kwh.mean, e1.discomfort_kh.mean, 6pt, 0pt, left + horizon,
        text(fill: group-colors(d.groups.epochs).at(0), weight: "bold")[1 epoch]),
      ..point(refs.oracle.energy_kwh, refs.oracle.discomfort_kh, "d", oracle-c, size: 6pt),
      at(refs.oracle.energy_kwh, refs.oracle.discomfort_kh, 0pt, -6pt, center + bottom,
        text(weight: "bold")[oracle]),
      ..point(refs.thermostat.energy_kwh, refs.thermostat.discomfort_kh, "d", thermostat-c, size: 6pt),
    )
  }

  let zoomed = {
    let iso = ()
    let xs = lq.linspace(x0, x1, num: 40)
    for (c, label) in iso-costs {
      iso.push(lq.plot(xs, xs.map(x => (c - x) / w), mark: none, stroke: iso-stroke, z-index: 0.5))
      // label where the line leaves the panel at the top or the left
      let (lx, ly) = if c - w * y1 >= x0 { (c - w * y1, y1) } else { (x0, (c - x0) / w) }
      iso.push(at(lx, ly, 2pt, 2pt, left + top, tag(fill: white.transparentize(15%), color: muted, inset: (x: 1pt, y: 1pt))[cost #label]))
    }
    lq.diagram(
      width: 100%,
      height: 5.2cm,
      title: panel("b", [Where each setting ends up]),
      xlim: (x0, x1),
      ylim: (y0, y1),
      xlabel: [Energy (kWh/day)],
      yaxis: (ticks: (0, 0.1, 0.2, 0.3, 0.4).map(t => (t, fmt(t, digits: 1)))),
      ..iso,
      lq.plot(pe, pd, color: baseline-c, stroke: path-stroke, mark: "o", mark-size: 2.4pt, z-index: 2),
      ..endpoints(true),
      ..point(bf.energy_kwh.mean, bf.discomfort_kh.mean, "o", baseline-c, size: 9pt, z: 6),
      at(bf.energy_kwh.mean, bf.discomfort_kh.mean, -7pt, 4pt, right + top,
        text(weight: "bold")[baseline, end]),
      ..point(refs.thermostat.energy_kwh, refs.thermostat.discomfort_kh, "d", thermostat-c, size: 8pt),
      at(refs.thermostat.energy_kwh, refs.thermostat.discomfort_kh, 0pt, 6pt, center + top,
        text(fill: thermostat-c.darken(10%), weight: "bold")[tuned thermostat]),
      lq.place(x1, y0, align: right + bottom, pad(right: 1pt, bottom: 2pt,
        tag(fill: white.transparentize(15%), color: muted, inset: (x: 1pt, y: 1pt))[baseline path, training runs #sym.arrow.l])),
    )
  }

  if title {
    heading-block(width: width, [Trading energy for comfort],
      sub: [End points are means of #n-seeds seeds. Most settings end on or near the path the baseline itself takes
        during training. Blues: values below the baseline; oranges and reds: above it.])
  }
  block(width: width, below: 0.7em, legend-row(gap: 1.1em,
    (box[#glyph("s")#glyph("^")#glyph("o")], [vary n_epochs, clip range $epsilon$, GAE $lambda$]),
    (swatch(color: baseline-c, thickness: 1.5pt, mark: dot(baseline-c, size: 3pt, stroke: none)),
      [baseline during training]),
    (swatch(color: muted.transparentize(40%), thickness: 1.2pt), [95% CI]),
    (swatch(color: muted, thickness: 0.8pt, dash: (0.5pt, 2.2pt)), [equal cost (energy + #fmt(w, digits: 0) × discomfort)]),
  ))
  show: lq.layout
  block(width: width, std.grid(columns: (1fr, 1.6fr), column-gutter: 0.8cm, overview, zoomed))
}
