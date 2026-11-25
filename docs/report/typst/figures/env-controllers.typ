// A cold and a hot day: HVAC off, a thermostat switched on at 8:00, and the oracle.
#import "style.typ": *

#let d = json("data/env.json")
#let cfg = d.config
#let days = d.controllers.days

#let keys = ("off", "thermostat", "oracle")
#let styles = (
  off: (color: off-c, stroke: (paint: off-c, thickness: 1.3pt, dash: (4pt, 2.5pt)), name: [HVAC off]),
  thermostat: (color: thermostat-c, stroke: (paint: thermostat-c, thickness: 1.4pt, join: "round"),
    name: [thermostat, on at 8:00]),
  oracle: (color: oracle-c, stroke: (paint: oracle-c, thickness: 2pt, join: "round"), name: [oracle]),
)
#let names = (cold: [Cold day], hot: [Hot day])
#let walls-stroke = (paint: mass-c, thickness: 1.3pt, dash: (0.4pt, 2.2pt), cap: "round")

#let render(width: text-width, title: true) = {
  show: theme
  let occ = (cfg.occupied_start, cfg.occupied_end)
  let occupied = lq.rect(occ.at(0), 0%, width: occ.at(1) - occ.at(0), height: 100%, fill: soft, stroke: none,
    z-index: 0)
  let hour-ticks(labels) = (ticks: range(0, 25, step: 4).map(h => (h, if labels [#h:00] else [])))

  // energy and discomfort of the day, per controller
  let scores(day) = box(fill: soft, radius: 2pt, inset: (x: 6pt, y: 5pt), {
    let m = day.metrics
    std.grid(
      columns: (auto, auto, auto),
      column-gutter: 0.9em,
      row-gutter: 0.45em,
      align: (left, right, right),
      text(weight: "bold")[Score of this day], note[energy], note[discomfort],
      std.grid.hline(stroke: 0.5pt + faint),
      ..keys.map(k => (
        box[#swatch(color: styles.at(k).color, dash: styles.at(k).stroke.at("dash", default: none), width: 1.1em,
          thickness: 1.5pt) #h(0.25em) #styles.at(k).name],
        [#fmt(m.at(k).energy) kWh],
        if k == "oracle" { text(weight: "bold")[#fmt(m.at(k).discomfort) K·h] } else [#fmt(m.at(k).discomfort) K·h],
      )).flatten(),
    )
  })

  let upper(day, letter) = {
    let all = day.t_out + day.t_air_off + day.t_air_oracle + day.t_air_thermostat
    let lo = calc.floor(calc.min(..all) / 5) * 5 - 1
    let hi = calc.ceil(calc.max(..all) / 5) * 5 + 1
    lq.diagram(
      width: 100%,
      height: 4.0cm,
      title: panel(letter, [#names.at(day.label)], unit: [mean outdoor temperature #fmt(day.t_out_mean) °C]),
      xlim: (0, 24),
      ylim: (lo, hi),
      xaxis: hour-ticks(false),
      yaxis: (ticks: range(int(lo) + 1, int(hi), step: 5).map(v => (v, [#fmt(v, digits: 0) °C]))),
      occupied,
      lq.rect(occ.at(0), cfg.comfort_low, width: occ.at(1) - occ.at(0), height: cfg.comfort_high - cfg.comfort_low,
        fill: comfort.transparentize(75%), stroke: none, z-index: 1),
      lq.plot(day.hours, day.t_out, color: outdoor-c, stroke: (paint: outdoor-c, thickness: 1pt, join: "round"),
        mark: none, z-index: 2),
      lq.plot(day.hours, day.t_mass_oracle, color: mass-c, stroke: walls-stroke, mark: none, z-index: 3),
      ..keys.map(k => lq.plot(day.hours, day.at("t_air_" + k), color: styles.at(k).color,
        stroke: styles.at(k).stroke, mark: none, z-index: if k == "oracle" { 5 } else { 4 })),
    )
  }

  let lower(day) = lq.diagram(
    width: 100%,
    height: 1.65cm,
    xlim: (0, 24),
    ylim: (-1.2, 1.2),
    xlabel: [Hour of day],
    xaxis: hour-ticks(true),
    yaxis: (ticks: ((-1, [−1]), (0, [0]), (1, [+1]))),
    grid: none,
    lq.rect(0, 0, width: 24, height: 1.2, fill: heat-soft, stroke: none, z-index: 0),
    lq.rect(0, -1.2, width: 24, height: 1.2, fill: cool-soft, stroke: none, z-index: 0),
    lq.place(0.3, 1.1, align: left + top, text(fill: heat, weight: "regular")[heating]),
    lq.place(0.3, -1.1, align: left + bottom, text(fill: cool, weight: "regular")[cooling]),
    lq.hlines(0, stroke: 0.5pt + subtle),
    lq.fill-between(day.hours, day.action_thermostat, step: start, fill: thermostat-c.transparentize(55%),
      stroke: none, z-index: 2),
    lq.plot(day.hours, day.action_oracle, step: start, mark: none, color: oracle-c,
      stroke: (paint: oracle-c, thickness: 1.6pt, join: "round"), z-index: 3),
  )


  if title {
    heading-block(width: width, [Starting late costs comfort; the oracle preheats],
      sub: [Two of the test days. The thermostat only reacts once people arrive at 8:00; the oracle knows the
        whole day in advance. Comfort is required in the green band, during the occupied hours (grey).])
  }
  block(width: width, below: 0.75em, legend-row(gap: 1.1em,
    ..keys.map(k => (swatch(color: styles.at(k).color, dash: styles.at(k).stroke.at("dash", default: none),
      thickness: 1.6pt), [air: #styles.at(k).name])),
    (swatch(color: mass-c, thickness: 1.4pt, dash: (0.4pt, 2.2pt)), [walls: oracle]),
    (swatch(color: outdoor-c, thickness: 1pt), [outdoors]),
    (swatch(kind: "area", color: comfort.transparentize(75%)), [comfort band]),
  ))
  show: lq.layout
  block(width: width, std.grid(
    columns: (1fr, 1fr),
    column-gutter: 0.9cm,
    row-gutter: 0.3cm,
    upper(days.at(0), "a"), upper(days.at(1), "b"),
    lower(days.at(0)), lower(days.at(1)),
    scores(days.at(0)), scores(days.at(1)),
  ))
}
