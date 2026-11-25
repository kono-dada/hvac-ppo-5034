// Learning curves of the ablation: evaluation on the 100 test days during training.
#import "style.typ": *

#let d = json("data/ablation.json")
#let n-seeds = d.configs.baseline.seeds.len()
#let panels = (
  (key: "cost", title: [Cost], unit: none, lim: (4.0, 9.0), ticks: (4, 5, 6, 7, 8, 9), digits: 0),
  (key: "energy_kwh", title: [Energy], unit: [kWh/day], lim: (3.5, 9.0), ticks: (4, 5, 6, 7, 8, 9), digits: 0),
  (key: "discomfort_kh", title: [Discomfort], unit: [K·h/day], lim: (0, 0.32), ticks: (0, 0.1, 0.2, 0.3), digits: 1),
)
#let group-order = ("epochs", "clip", "lambda")

/// Legend entries of one ablation group, in one line: a coloured value per setting, baseline in bold.
/// (Also used by ablation-update-size.typ.)
#let group-legend(g, colors, extra: ()) = {
  let items = g.values.zip(g.names, colors).map(((v, n, c)) => box[
    #swatch(color: c, thickness: if n == "baseline" { 2pt } else { 1.5pt }, width: 1.1em)#h(0.25em)#if n == "baseline" [*#v*] else [#v]])
  (items + extra).join(h(0.75em))
}

#let render(width: text-width, title: true) = {
  show: theme
  let refs = d.refs
  let k = d.total_steps / 1000
  if title {
    heading-block(width: width, [Learning curves: evaluation on the 100 test days during training],
      sub: [Line: mean of #n-seeds seeds; band: ±1 standard deviation across seeds. Blues: values below the
        baseline (black), oranges and reds: above it. Curves start at the first evaluation after training
        begins (8k steps); early values above the plotted range are cut off.])
  }
  // reference lines and their key
  let ref-lines(p) = (
    lq.hlines(refs.thermostat.at(p.key), stroke: (paint: thermostat-c, thickness: 0.9pt, dash: (3pt, 2pt)), z-index: 2),
    lq.hlines(refs.oracle.at(p.key), stroke: (paint: muted, thickness: 0.9pt, dash: (0.5pt, 1.8pt), cap: "round"),
      z-index: 2),
  )
  block(width: width, below: 0.7em, legend-row(gap: 1.2em,
    (swatch(kind: "area", color: ink.transparentize(85%), width: 1.2em), [±1 s.d. across seeds]),
    (swatch(color: thermostat-c, thickness: 1pt, dash: (3pt, 2pt)), [tuned thermostat]),
    (swatch(color: muted, thickness: 1pt, dash: (0.5pt, 1.8pt)), [oracle (knows the weather in advance)]),
  ))
  let cells = ()
  for (row, gname) in group-order.enumerate() {
    let g = d.groups.at(gname)
    let colors = group-colors(g)
    // row header: the ablated setting and its values
    cells.push(std.grid.cell(colspan: 3, {
      if row > 0 { v(0.12cm); line(length: 100%, stroke: 0.5pt + faint); v(-0.05cm) }
      text(weight: "bold", group-title.at(gname))
      h(0.9em)
      group-legend(g, colors)
    }))
    for (col, p) in panels.enumerate() {
      let plots = ()
      // non-baseline first, baseline on top
      let order = range(g.names.len()).sorted(key: i => if g.names.at(i) == "baseline" { 1 } else { 0 })
      for i in order {
        let name = g.names.at(i)
        let c = d.configs.at(name)
        // skip the evaluation at step 0: the untrained policy barely heats (discomfort ~35 K·h)
        let cv = c.curves.at(p.key)
        let (m, sd) = (cv.mean.slice(1), cv.sd.slice(1))
        let xs = c.t.slice(1).map(t => t / 1000)
        let base = name == "baseline"
        plots.push(lq.fill-between(xs, m.zip(sd).map(((a, b)) => a - b),
          y2: m.zip(sd).map(((a, b)) => a + b), fill: colors.at(i).transparentize(if base { 90% } else { 91% }),
          z-index: 1))
        plots.push(lq.plot(xs, m, color: colors.at(i), mark: none,
          stroke: (paint: colors.at(i), thickness: if base { 1.8pt } else { 1.3pt }, join: "round"),
          z-index: if base { 4 } else { 3 }))
      }
      cells.push(lq.diagram(
        width: 100%,
        height: 2.5cm,
        title: if row == 0 { panel("abc".at(col), p.title, unit: p.unit) },
        xlim: (0, k),
        ylim: p.lim,
        xlabel: if row == 2 [Training steps (thousands)],
        xaxis: (ticks: (0, 50, 100, 150, 200).map(t => (t, if row == 2 [#t] else []))),
        yaxis: (ticks: p.ticks.map(t => (t, fmt(t, digits: p.digits)))),
        ..ref-lines(p),
        ..plots,
      ))
    }
  }
  show: lq.layout
  block(width: width, std.grid(columns: (1fr, 1fr, 1fr), column-gutter: 0.7cm, row-gutter: 0.22cm, ..cells))
}
