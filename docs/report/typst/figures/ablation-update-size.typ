// How large each PPO update is: approx_kl and clip_fraction per update, mean of 10 seeds (log axis).
#import "style.typ": *

#let d = json("data/ablation.json")
#let n-seeds = d.configs.baseline.seeds.len()
#let biased = ("epochs_1", "epochs_3") // approx_kl only covers the last pass of each update for these
#let half = 2                          // centred moving average over 2 * half + 1 updates
#let rows = (
  (key: "approx_kl", label: [approx_kl], lim: (1e-4, 0.5), floor: 1e-4,
    ticks: ((1e-4, [0.0001]), (1e-3, [0.001]), (1e-2, [0.01]), (0.1, [0.1]))),
  (key: "clip_fraction", label: [clip_fraction], lim: (3e-3, 0.5), floor: 3e-3,
    ticks: ((0.01, [1%]), (0.03, [3%]), (0.1, [10%]), (0.3, [30%]))),
)
#let cols = ("clip", "epochs")

/// Centred moving average that skips missing values; values are clamped to `floor` (log axis).
#let smooth(ys, floor) = range(ys.len()).map(i => {
  let w = ys.slice(calc.max(0, i - half), calc.min(ys.len(), i + half + 1)).filter(y => y != none)
  if w.len() == 0 or ys.at(i) == none { float.nan } else { calc.max(floor, w.sum() / w.len()) }
})

#let render(width: text-width, title: true) = {
  show: theme
  if title {
    heading-block(width: width, [How large each update is],
      sub: [Mean of #n-seeds seeds per update, smoothed over #(2 * half + 1) updates; logarithmic vertical axis.
        Black and bold: the baseline setting. Dashed: with 1 and 3 epochs, approx_kl only measures the last pass
        of each update and is therefore too small (section 3.1).])
  }
  let cells = ()
  for (r, row) in rows.enumerate() {
    for (c, gname) in cols.enumerate() {
      let g = d.groups.at(gname)
      let colors = group-colors(g)
      let objs = ()
      let ends = ()
      for (i, name) in g.names.enumerate() {
        let cf = d.configs.at(name)
        let ys = smooth(cf.at(row.key), row.floor)
        let base = name == "baseline"
        let dashed = row.key == "approx_kl" and name in biased
        objs.push(lq.plot(cf.iterations, ys, mark: none, color: colors.at(i),
          stroke: (paint: colors.at(i), thickness: if base { 1.8pt } else { 1.3pt }, join: "round",
            dash: if dashed { (3pt, 1.8pt) } else { none }),
          z-index: if base { 4 } else { 3 }))
        ends.push((y: calc.log(ys.last()), name: name, value: g.values.at(i), color: colors.at(i), base: base,
          dashed: dashed))
      }
      // direct labels at the right end, spread apart on the log axis
      let span = calc.log(row.lim.at(1)) - calc.log(row.lim.at(0))
      let spread = dodge(ends.map(e => e.y), 0.085 * span, lo: calc.log(row.lim.at(0)) + 0.03 * span,
        hi: calc.log(row.lim.at(1)) - 0.03 * span)
      for (e, y) in ends.zip(spread) {
        let label = [#e.value]
        objs.push(lq.place(101.5, calc.pow(10, y), align: left + horizon,
          text(fill: e.color, size: 0.95em, weight: if e.base { "bold" } else { "medium" }, label)))
      }
      objs.push(lq.place(1, 0%, align: left + top, pad(left: 1pt, top: 1pt, text(weight: "semibold", row.label))))
      if r == 0 and gname == "clip" {
        let lo = d.summary.at("clip_0.05").approx_kl.mean
        let hi = d.summary.at("clip_0.5").approx_kl.mean
        objs.push(lq.place(99, 100%, align: right + bottom, pad(bottom: 3pt, box(fill: white.transparentize(15%),
          inset: 2pt, align(right, note[mean over training #h(0.2em) *#sym.times#calc.round(hi / lo)* \
            #fmt(lo, digits: 4) at $epsilon$ = 0.05, #fmt(hi, digits: 3) at 0.5])))))
      }
      if r == 0 and gname == "epochs" {
        objs.push(lq.place(99, 100%, align: right + bottom, pad(bottom: 3pt, box(fill: white.transparentize(15%),
          inset: 2pt, align(right, note[dashed: measures the last pass \ only, so too small])))))
      }
      cells.push(lq.diagram(
        width: 100%,
        height: 4.1cm,
        title: if r == 0 {
          panel("ab".at(c), [Varying #group-title.at(gname)])
        },
        xlim: (1, 116),
        ylim: row.lim,
        yscale: "log",
        xlabel: if r == 1 [Update number],
        xaxis: (ticks: (1, 25, 50, 75, 100).map(t => (t, if r == 1 [#t] else []))),
        yaxis: (ticks: row.ticks),
        ..objs,
      ))
    }
  }
  show: lq.layout
  block(width: width, std.grid(columns: (1fr, 1fr), column-gutter: 0.75cm, row-gutter: 0.35cm, ..cells))
}
