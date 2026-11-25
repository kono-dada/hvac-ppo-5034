// Twelve random days: outdoor temperature and solar gain, coloured by the day's mean temperature.
#import "style.typ": *

#let d = json("data/env.json")
#let cfg = d.config
#let days = d.days

// cold -> mild -> hot; dark enough at every point to read as a thin line on white
#let day-map = gradient.linear(
  rgb("#1f4f8f"), rgb("#4a8cc6"), rgb("#9b9d94"), rgb("#e38a35"), rgb("#b02a1f"), space: oklab,
)
#let (lo, hi) = cfg.t_out_mean_range
#let frac(t) = calc.clamp((t - lo) / (hi - lo), 0, 1)
#let day-color(t) = day-map.sample(frac(t) * 100%)

/// Horizontal colour bar with a tick for every drawn day.
#let colour-bar(w) = {
  let h = 6pt
  box(width: w, height: h + 13pt, {
    place(top + left, dy: 3pt, rect(width: w, height: h, radius: 1pt, fill: gradient.linear(..day-map.stops(),
      space: oklab)))
    for day in days.days {
      place(top + left, dx: w * frac(day.t_out_mean) - 0.4pt, line(angle: 90deg, length: h + 3pt,
        stroke: 0.8pt + ink))
    }
    for t in range(0, int(hi) + 1, step: 10) {
      place(top + left, dx: w * frac(t), dy: h + 5pt, place(center + top, text(fill: muted)[#t]))
    }
  })
}

#let render(width: text-width, title: true) = {
  show: theme
  let occ = (cfg.occupied_start, cfg.occupied_end)
  let occupied(body) = (
    lq.rect(occ.at(0), 0%, width: occ.at(1) - occ.at(0), height: 100%, fill: soft, stroke: none, z-index: 0),
    lq.place((occ.at(0) + occ.at(1)) / 2, 100%, align: bottom, pad(bottom: 3pt, note(body))),
  )
  let common = (
    width: 100%,
    height: 4.7cm,
    xlim: (0, 24),
    xlabel: [Hour of day],
    xaxis: (ticks: range(0, 25, step: 4).map(h => (h, [#h:00]))),
  )
  let lines(key) = days.days.sorted(key: day => calc.abs(day.t_out_mean - 15)).map(day => lq.plot(days.hours,
    day.at(key), color: day-color(day.t_out_mean), stroke: (thickness: 1.25pt, join: "round"), mark: none,
    z-index: 3))
  let n = days.days.len()

  if title {
    heading-block(width: width, [Every episode is a different day],
      sub: [#n days drawn at random. Each day keeps its colour in both panels: its mean outdoor temperature,
        drawn uniformly between #fmt(lo, digits: 0) and #fmt(hi, digits: 0) °C.])
  }
  // colour key
  block(width: width, below: 0.7em, {
    box(baseline: -2pt)[Daily mean outdoor temperature (°C)]
    h(0.8em)
    box(colour-bar(5.2cm), baseline: 9pt)
    h(0.5em)
    box(baseline: -2pt, note[ticks: the #n days shown])
  })
  show: lq.layout
  block(width: width, std.grid(
    columns: (1.5fr, 1fr),
    column-gutter: 0.9cm,
    lq.diagram(
      ..common,
      title: panel("a", [Outdoor temperature], unit: [°C]),
      ylim: (-4.5, 37.5),
      yaxis: (ticks: range(0, 40, step: 10)),
      ..occupied[occupied #int(occ.at(0))–#int(occ.at(1)) h],
      lq.rect(occ.at(0), cfg.comfort_low, width: occ.at(1) - occ.at(0), height: cfg.comfort_high - cfg.comfort_low,
        fill: comfort.transparentize(78%), stroke: none, z-index: 1),
      ..lines("t_out"),
      lq.place(occ.at(0), (cfg.comfort_low + cfg.comfort_high) / 2, align: right + horizon, pad(right: 3pt,
        tag(fill: white.transparentize(10%), color: comfort.darken(15%), weight: "bold")[comfort
          #int(cfg.comfort_low)–#int(cfg.comfort_high) °C])),
    ),
    lq.diagram(
      ..common,
      xaxis: (ticks: range(0, 25, step: 6).map(h => (h, [#h:00]))),
      title: panel("b", [Sun through the window], unit: [W]),
      ylim: (0, cfg.q_solar_peak * 1.05),
      yaxis: (ticks: range(0, int(cfg.q_solar_peak) + 1, step: 250), exponent: 0),
      ..occupied[occupied],
      ..lines("q_solar"),
    ),
  ))
}
