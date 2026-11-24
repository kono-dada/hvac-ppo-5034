// Step response of the two-node room: two hours of full heating, then off.
#import "style.typ": *

#let d = json("data/env.json")
#let s = d.step
#let cfg = d.config

#let render(width: text-width, title: true) = {
  show: theme
  let (hours, air, mass) = (s.hours, s.air, s.mass)
  let (io, il) = (s.i_off, s.i_later)
  let n = hours.len() - 1
  let (tau-fast, tau-slow) = cfg.time_constants_h
  let dot-at(x, y, color) = lq.plot((x,), (y,), stroke: none, mark: "o", mark-size: 5.5pt, color: color, z-index: 6)
  let label-at(x, y, body, align: left + horizon, pad-args: (:)) = lq.place(x, y, align: align,
    pad(..pad-args, text(size: 0.98em, body)))

  if title {
    heading-block(width: width, [The air responds within the hour; the room as a whole takes days],
      sub: [Room and outdoors start at #fmt(s.t_out, digits: 0) °C. The heat pump runs at full power
        (#fmt(cfg.q_hvac_max / 1000, digits: 0) kW of heat) from hour #fmt(s.on_from, digits: 0) to hour
        #fmt(s.on_to, digits: 0), then switches off.])
  }
  block(width: width, lq.diagram(
    width: 100%,
    height: 5.4cm,
    xlim: (0, s.total),
    ylim: (14.0, 23.6),
    xlabel: [Hours since the start],
    xaxis: (ticks: range(0, int(s.total) + 1)),
    yaxis: (ticks: range(15, 24, step: 2).map(v => (v, [#v °C]))),
    // heating period: tinted column with a solid bar on top
    lq.rect(s.on_from, 0%, width: s.on_to - s.on_from, height: 100%, fill: heat-soft, stroke: none, z-index: 0),
    lq.rect(s.on_from, 0%, width: s.on_to - s.on_from, height: 3pt, fill: heat, stroke: none, z-index: 1),
    lq.place((s.on_from + s.on_to) / 2, 0%, align: top, pad(top: 6pt, align(center,
      text(fill: heat, size: 0.98em)[*heat pump on* \ full power]))),
    // outdoors
    lq.hlines(s.t_out, stroke: (paint: outdoor-c, thickness: 1pt, dash: (3pt, 2pt))),
    lq.place(s.total, s.t_out, align: right + top, pad(top: 3pt, text(fill: outdoor-dark, size: 0.96em)[outdoors
      #fmt(s.t_out, digits: 0) °C])),
    // the two temperatures
    lq.plot(hours, mass, color: mass-c, stroke: (thickness: 2pt, join: "round"), mark: none, z-index: 4),
    lq.plot(hours, air, color: air-c, stroke: (thickness: 2pt, join: "round"), mark: none, z-index: 5),
    // switch-off
    dot-at(hours.at(io), air.at(io), air-c),
    label-at(hours.at(io), air.at(io), pad-args: (left: 7pt),
      text(fill: air-c)[*Air* $T_a$ #h(0.3em) *#fmt(air.at(io)) °C* when the heat pump stops]),
    dot-at(hours.at(io), mass.at(io), mass-c),
    label-at(hours.at(io), mass.at(io), align: left + top, pad-args: (left: 6pt, top: 4pt),
      text(fill: mass-c)[*Walls & floor* $T_m$ #h(0.3em) only *#fmt(mass.at(io)) °C*]),
    // one hour later
    dot-at(hours.at(il), air.at(il), air-c),
    label-at(hours.at(il), air.at(il), align: left + bottom, pad-args: (left: 5pt, bottom: 4pt),
      text(fill: air-c)[*#fmt(air.at(il)) °C* one hour later: the air falls back towards the walls]),
    // long after
    label-at(hours.at(n), (air.at(n) + mass.at(n)) / 2 - 0.25, align: right + top,
      text(fill: ink-2)[#fmt(hours.at(n) - s.on_to, digits: 0) h after switch-off: both still
        *#fmt((air.at(n) + mass.at(n)) / 2) °C*]),
    // the two time scales of the model
    lq.place(s.total, 0%, align: right + top, pad(top: 4pt, box(fill: white, inset: (x: 5pt, y: 4pt), radius: 2pt,
      stroke: 0.5pt + faint, align(left, text(size: 0.95em)[
        #text(weight: "semibold")[Two time scales of the model] \
        #text(fill: muted)[air follows the heat pump:] *#fmt(tau-fast) h* \
        #text(fill: muted)[whole room follows the outdoors:] *#fmt(tau-slow, digits: 0) h*
      ])))),
  ))
}
