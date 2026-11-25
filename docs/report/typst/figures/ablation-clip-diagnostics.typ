// Why the clip range matters although few samples are clipped: update size and clip fraction over
// training (10 seeds), and the post-hoc gradient diagnostics late in training (rounds 71-100, 3 seeds).
#import "style.typ": *
#import "ablation-learning-curves.typ": group-legend

#let d = json("data/ablation.json")
#let diag = json("../data/report.json").diagnostics.late
#let g = d.groups.clip
#let n-seeds = d.configs.baseline.seeds.len()
#let half = 2  // centred moving average over 2 * half + 1 updates

/// Centred moving average that skips missing values; values are clamped to `floor` (log axis).
#let smooth(ys, floor) = range(ys.len()).map(i => {
  let w = ys.slice(calc.max(0, i - half), calc.min(ys.len(), i + half + 1)).filter(y => y != none)
  if w.len() == 0 or ys.at(i) == none { float.nan } else { calc.max(floor, w.sum() / w.len()) }
})

#let render(width: text-width, title: true) = {
  show: theme
  let colors = group-colors(g)
  if title {
    heading-block(width: width, [The clip range sets the size of an update, although few samples are clipped],
      sub: [a, b: every update, mean of #n-seeds seeds, smoothed over #(2 * half + 1) updates.
        c, d: post-hoc diagnostics, updates 71–100, 3 seeds.])
  }
  // non-baseline first, baseline on top
  let order = range(g.names.len()).sorted(key: i => if g.names.at(i) == "baseline" { 1 } else { 0 })
  let line-style(i) = (paint: colors.at(i), thickness: if g.names.at(i) == "baseline" { 1.8pt } else { 1.3pt },
    join: "round")
  let over-training(key, floor) = order.map(i => {
    let c = d.configs.at(g.names.at(i))
    lq.plot(c.iterations, smooth(c.at(key), floor), mark: none, color: colors.at(i), stroke: line-style(i),
      z-index: if g.names.at(i) == "baseline" { 4 } else { 3 })
  })
  let log-ticks = ((1e-3, [0.001]), (1e-2, [0.01]), (0.1, [0.1]))
  let update-ticks = (1, 50, 100).map(t => (t, [#t]))

  let a = lq.diagram(
    width: 100%, height: 3.0cm,
    title: panel("a", [`approx_kl`]),
    xlim: (1, 100), ylim: (2e-4, 0.4), yscale: "log",
    xlabel: [Update], xaxis: (ticks: update-ticks), yaxis: (ticks: log-ticks),
    ..over-training("approx_kl", 2e-4),
  )
  let b = lq.diagram(
    width: 100%, height: 3.0cm,
    title: panel("b", [`clip_fraction`]),
    xlim: (1, 100), ylim: (0, 0.25),
    xlabel: [Update], xaxis: (ticks: update-ticks),
    yaxis: (ticks: (0, 0.1, 0.2).map(t => (t, [#calc.round(100 * t)%]))),
    ..over-training("clip_fraction", 0),
  )
  // c: per clip range, share of samples whose gradient the clipping removes, and the length of their
  //    gradient relative to the unclipped gradient
  let xs = range(g.values.len())
  let at(key) = g.values.map(v => diag.at(str(v)).at(key))
  let series(key, color, label) = (
    lq.plot(xs, at(key), color: color, mark: "o", mark-size: 3.5pt, stroke: (paint: color, thickness: 1.1pt)),
    lq.place(2.1, at(key).at(2) + 0.04, align: left + bottom, text(fill: color, label)),
  )
  let c = lq.diagram(
    width: 100%, height: 3.0cm,
    title: panel("c", [Zeroed share]),
    xlim: (-0.4, 4.4), ylim: (0, 1.0),
    xlabel: [Clip range $epsilon$], xaxis: (ticks: xs.zip(g.values).map(((x, v)) => (x, [#v]))),
    yaxis: (ticks: (0, 0.25, 0.5, 0.75, 1).map(t => (t, [#calc.round(100 * t)%]))),
    ..series("z_norm_share", ink, [gradient]),
    ..series("zeroed_frac", muted, [samples]),
  )
  // d: KL to the data-collecting policy after each pass over the batch
  let epochs = range(1, 11)
  let dd = lq.diagram(
    width: 100%, height: 3.0cm,
    title: panel("d", [KL per pass]),
    xlim: (0.6, 10.4), ylim: (8e-4, 0.25), yscale: "log",
    xlabel: [Pass over the batch], xaxis: (ticks: (1, 5, 10).map(t => (t, [#t]))), yaxis: (ticks: log-ticks),
    ..order.map(i => lq.plot(epochs, diag.at(str(g.values.at(i))).kl_by_epoch, mark: "o", mark-size: 2.6pt,
      color: colors.at(i), stroke: line-style(i), z-index: if g.names.at(i) == "baseline" { 4 } else { 3 })),
  )
  block(width: width, below: 0.6em, {
    text(weight: "bold", group-title.clip)
    h(0.9em)
    group-legend(g, colors)
  })
  show: lq.layout
  block(width: width, std.grid(columns: (1fr,) * 4, column-gutter: 0.6cm, a, b, c, dd))
}
