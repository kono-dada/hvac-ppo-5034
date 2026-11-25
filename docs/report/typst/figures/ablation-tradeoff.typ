// Energy versus discomfort: every configuration's end point on top of the baseline's own training path.
//   a  the whole path of the baseline (from 16k steps; earlier points are far off the scale),
//      with its start, its end and arrows in the direction of training
//   b  zoomed in on the end: where every setting ends up, with 95% intervals and lines of equal cost
#import "style.typ": *
#import "@preview/tiptoe:0.4.0"

#let d = json("data/ablation.json")
#let w = d.weight
#let n-seeds = d.configs.baseline.seeds.len()
#let shape = (epochs: "s", clip: "^", lambda: "o")
#let label-of(g, v) = if g == "epochs" { if v == 1 [1 epoch] else [#v epochs] } else if g == "clip" [$epsilon$ #v] else [$lambda$ #v]

// first step shown: before it the discomfort is far off the scale (35 K·h/day at step 0)
#let first-step = 16384

// panel b: where each label sits relative to its point (dx, dy, alignment of the label).
// Labels go diagonally off their point so they cover neither the point nor its interval arms.
#let offsets = (
  "epochs_3": (4pt, -3pt, left + bottom),
  "epochs_30": (-4pt, -3pt, right + bottom),
  "clip_0.05": (4pt, 3pt, left + top),
  "clip_0.1": (3pt, 4pt, left + top),
  "clip_0.3": (4pt, -3pt, left + bottom),
  "clip_0.5": (4pt, -3pt, left + bottom),
  "lambda_0.8": (-4pt, -3pt, right + bottom),
  "lambda_0.9": (4pt, -3pt, left + bottom),
  "lambda_0.99": (0pt, 5pt, center + top),
  "lambda_1": (4pt, -3pt, left + bottom),
)

/// A filled marker with a white outline.
#let point(x, y, m, color, size: 7pt, z: 5) = (
  lq.plot((x,), (y,), mark: m, mark-size: size + 2.4pt, color: white, stroke: none, z-index: z),
  lq.plot((x,), (y,), mark: m, mark-size: size, color: color, stroke: none, z-index: z + 0.1),
)
/// A hollow ring (the start of the training path): a white disc inside a filled one.
#let ring(x, y, color, size: 7pt, z: 6) = (
  ..point(x, y, "o", color, size: size, z: z),
  lq.plot((x,), (y,), mark: "o", mark-size: size - 3pt, color: white, stroke: none, z-index: z + 0.2),
)
/// A label on a translucent white background, offset from a data point.
#let at(x, y, dx, dy, al, body, z: 7) = lq.place(x, y, align: al, z-index: z,
  move(dx: dx, dy: dy, box(fill: white.transparentize(15%), radius: 1pt, inset: (x: 1.2pt, y: 1pt), body)))

/// Inline glyphs for the legend.
#let glyph(kind, color: muted) = box(width: 0.8em, height: 0.7em, baseline: 0.02em, align(center + horizon, {
  if kind == "s" { rect(width: 5.4pt, height: 5.4pt, fill: color) }
  else if kind == "^" { polygon.regular(vertices: 3, size: 7.4pt, fill: color) }
  else if kind == "ring" { circle(radius: 2.6pt, fill: white, stroke: 1pt + color) }
  else { circle(radius: 3pt, fill: color) }
}))

#let render(width: text-width, title: true) = {
  show: theme
  let base = d.configs.baseline
  let bf = base.final
  let keep = base.t.enumerate().filter(((i, t)) => t >= first-step).map(((i, _)) => i)
  let pt-at(i) = (base.curves.energy_kwh.mean.at(i), base.curves.discomfort_kh.mean.at(i))
  let path = keep.map(pt-at)
  let pe = path.map(p => p.at(0))
  let pd = path.map(p => p.at(1))
  let (start-e, start-d) = path.first()
  let step-k(k) = [#calc.round(k / 1000)k]
  let refs = d.refs
  let (x0, x1, y0, y1) = (4.0, 5.45, -0.015, 0.4)
  let path-stroke = (paint: baseline-c, thickness: 1.4pt, join: "round", cap: "round")
  let iso-stroke = (paint: muted, thickness: 0.7pt, dash: (0.5pt, 2.2pt), cap: "round")
  let ci-stroke(c) = 0.9pt + c.transparentize(55%)
  let small-tag(body) = tag(fill: white.transparentize(15%), color: muted, inset: (x: 1pt, y: 1pt), body)
  let iso-costs = (
    (refs.thermostat.cost, [cost #fmt(refs.thermostat.cost, digits: 2) (thermostat)]),
    (bf.cost.mean, [cost #fmt(bf.cost.mean, digits: 2) (baseline)]),
    (5.0, [cost 5.0]),
    (5.5, [cost 5.5]),
  )

  // Arrowheads in the direction of training: on the path segment from step k to the next one,
  // the head sits at `frac` of the way along it.
  let arrow(k, frac: 0.6) = {
    let j = base.t.position(t => t == k)
    let (a, b) = (pt-at(j), pt-at(j + 1))
    let m = (a.at(0) + frac * (b.at(0) - a.at(0)), a.at(1) + frac * (b.at(1) - a.at(1)))
    lq.line(a, m, stroke: path-stroke, tip: tiptoe.stealth.with(length: 8pt, width: 6.8pt, inset: 25%), z-index: 2.2)
  }

  // end points of every setting with their 95% intervals (panel b only; 1 epoch lies outside it)
  let endpoints = {
    let out = ()
    for (gname, g) in d.groups {
      let colors = group-colors(g)
      for (i, name) in g.names.enumerate() {
        if name == "baseline" or name not in offsets { continue }
        let f = d.configs.at(name).final
        let (e, q) = (f.energy_kwh, f.discomfort_kh)
        let c = colors.at(i)
        out.push(lq.line((e.lo, q.mean), (e.hi, q.mean), stroke: ci-stroke(c), z-index: 3))
        out.push(lq.line((e.mean, q.lo), (e.mean, q.hi), stroke: ci-stroke(c), z-index: 3))
        let (dx, dy, al) = offsets.at(name)
        out.push(at(e.mean, q.mean, dx, dy, al, text(fill: c.darken(8%), weight: "bold",
          label-of(gname, g.values.at(i)))))
        out += point(e.mean, q.mean, shape.at(gname), c, size: 6.5pt)
      }
    }
    out
  }

  let height = 6.0cm

  let overview = {
    let step-label(k, dx, dy, al) = {
      let (x, y) = pt-at(base.t.position(t => t == k))
      at(x, y, dx, dy, al, note(step-k(k)))
    }
    let e1 = d.configs.epochs_1.final
    let c1 = group-colors(d.groups.epochs).at(0)
    lq.diagram(
      width: 100%,
      height: height,
      title: panel("a", [Whole training path]),
      ylabel: [Discomfort (K·h/day)],
      xlim: (3.1, 9.3),
      ylim: (-0.03, 0.5),
      xlabel: [Energy (kWh/day)],
      xaxis: (ticks: range(4, 10)),
      yaxis: (ticks: (0, 0.1, 0.2, 0.3, 0.4).map(t => (t, fmt(t, digits: 1)))),
      // the range of panel b
      lq.rect(x0, y0, width: x1 - x0, height: y1 - y0, fill: soft, stroke: none, z-index: 0),
      lq.place(x0, y1, align: left + bottom, pad(bottom: 1pt, small-tag[panel b])),
      // the baseline path, its direction, start and end
      lq.plot(pe, pd, color: baseline-c, stroke: path-stroke, mark: "o", mark-size: 2.2pt, z-index: 2),
      arrow(16384, frac: 0.5),
      arrow(40960, frac: 0.5),
      ..ring(start-e, start-d, baseline-c, size: 6.5pt),
      at(start-e, start-d, -6pt, 0pt, right + horizon,
        [*start* #note[(#step-k(first-step) steps)]]),
      ..point(bf.energy_kwh.mean, bf.discomfort_kh.mean, "o", baseline-c, size: 6.5pt, z: 6),
      step-label(32768, 0pt, -5pt, center + bottom),
      step-label(65536, 0pt, -5pt, center + bottom),
      // references outside panel b
      ..point(e1.energy_kwh.mean, e1.discomfort_kh.mean, "s", c1, size: 5.5pt),
      at(e1.energy_kwh.mean, e1.discomfort_kh.mean, 5pt, 0pt, left + horizon,
        text(fill: c1, weight: "bold")[1 epoch]),
      ..point(refs.oracle.energy_kwh, refs.oracle.discomfort_kh, "d", oracle-c, size: 6pt),
      at(refs.oracle.energy_kwh, refs.oracle.discomfort_kh, 0pt, -6pt, center + bottom,
        text(weight: "bold")[oracle]),
    )
  }

  let zoomed = {
    let iso = ()
    let xs = lq.linspace(x0, x1, num: 40)
    for (c, label) in iso-costs {
      iso.push(lq.plot(xs, xs.map(x => (c - x) / w), mark: none, stroke: iso-stroke, z-index: 0.5))
      // label where the line leaves the panel at the top or the left
      let (lx, ly) = if c - w * y1 >= x0 { (c - w * y1, y1) } else { (x0, (c - x0) / w) }
      iso.push(at(lx, ly, 2pt, 2pt, left + top, small-tag(label), z: 1))
    }
    lq.diagram(
      width: 100%,
      height: height,
      title: panel("b", [Where each setting ends up]),
      xlim: (x0, x1),
      ylim: (y0, y1),
      xlabel: [Energy (kWh/day)],
      xaxis: (ticks: range(0, 8).map(i => 4.0 + 0.2 * i).map(t => (t, fmt(t, digits: 1)))),
      yaxis: (ticks: (0, 0.1, 0.2, 0.3, 0.4).map(t => (t, fmt(t, digits: 1)))),
      ..iso,
      lq.plot(pe, pd, color: baseline-c, stroke: path-stroke, mark: "o", mark-size: 2.2pt, z-index: 2),
      arrow(73728, frac: 0.8),
      arrow(90112, frac: 0.45),
      ..endpoints,
      ..point(bf.energy_kwh.mean, bf.discomfort_kh.mean, "o", baseline-c, size: 8.5pt, z: 6),
      at(bf.energy_kwh.mean, bf.discomfort_kh.mean, -5pt, 6pt, right + top,
        [*baseline end* \ #note[(#step-k(base.t.last()) steps)]]),
      ..point(refs.thermostat.energy_kwh, refs.thermostat.discomfort_kh, "d", thermostat-c, size: 7.5pt),
      at(refs.thermostat.energy_kwh, refs.thermostat.discomfort_kh, 3pt, 6pt, right + top,
        align(right, text(fill: thermostat-c.darken(10%), weight: "bold")[tuned \ thermostat])),
    )
  }

  if title {
    heading-block(width: width, [Trading energy for comfort],
      sub: [End points are means of #n-seeds seeds. Most settings end on or near the path the baseline itself takes
        during training. Blues: values below the baseline; oranges and reds: above it.])
  }
  block(width: width, below: 0.8em, legend-row(gap: 0.95em,
    (glyph("s"), [n_epochs]),
    (glyph("^"), [clip range $epsilon$]),
    (glyph("o"), [GAE $lambda$]),
    (box[#glyph("ring", color: baseline-c)#h(0.15em)#swatch(color: baseline-c, thickness: 1.4pt, width: 1.4em,
        mark: text(fill: baseline-c, size: 7pt, sym.triangle.filled.r))#h(0.15em)#glyph("o", color: baseline-c)],
      [baseline during training]),
    (swatch(color: muted.transparentize(40%), thickness: 0.9pt), [95% CI]),
    (swatch(color: muted, thickness: 0.8pt, dash: (0.5pt, 2.2pt)), [equal cost (energy + #fmt(w, digits: 0) × discomfort)]),
  ))
  show: lq.layout
  block(width: width, std.grid(columns: (1fr, 1.75fr), column-gutter: 0.8cm, overview, zoomed))
}
