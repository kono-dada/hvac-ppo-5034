// Schematic of the two-node RC room model.
#import "style.typ": *
#import "@preview/cetz:0.5.3"

#let cfg = json("data/env.json").config

#let render(width: text-width, title: true) = {
  show: theme
  if title {
    heading-block(width: width, [The room as two heat stores],
      sub: [Heat flows along every arrow in proportion to the temperature difference; the agent only controls
        the heat pump.])
  }
  let scale = width / 16cm
  let (tau-fast, tau-slow) = cfg.time_constants_h
  let wall-line = rgb("#5d6670")
  cetz.canvas(length: 1cm * scale, {
    import cetz.draw: *
    let tip = (fill: auto, scale: 0.85)
    let arrow2(a, b, color: ink-2, thickness: 1.1pt) = line(a, b, stroke: (paint: color, thickness: thickness),
      mark: (start: "stealth", end: "stealth", ..tip))
    let arrow1(a, b, color: ink-2, thickness: 1.1pt) = line(a, b, stroke: (paint: color, thickness: thickness),
      mark: (end: "stealth", ..tip))
    // a conductance: value in a pill on the arrow, symbol and meaning above it
    let conductance(p, value, symbol, meaning) = {
      content(p, box(fill: white, stroke: 0.6pt + subtle, radius: 5pt, inset: (x: 5pt, y: 2.4pt),
        text(weight: "semibold", size: 0.98em, value)))
      content((p.at(0), p.at(1) + 0.36), anchor: "south", align(center, text(size: 0.95em)[
        #symbol #if meaning != none { linebreak(); text(fill: muted, meaning) }]))
    }
    let node(a, b, fill, stroke, head, cap, speed) = {
      rect(a, b, radius: 0.16, fill: fill, stroke: 1.3pt + stroke)
      let cx = (a.at(0) + b.at(0)) / 2
      content((cx, b.at(1) - 0.45), text(fill: stroke, weight: "bold", size: 1.12em, head))
      content((cx, b.at(1) - 1.0), text(fill: ink-2, size: 0.98em, cap))
      content((cx, a.at(1) + 0.38), text(fill: stroke, weight: "medium", size: 0.95em, speed))
    }

    // outdoors and the room
    rect((0, 0), (16, 7.7), radius: 0.3, fill: outdoor-c.transparentize(86%), stroke: none)
    content((0.35, 7.3), anchor: "west", text(fill: outdoor-dark, weight: "bold", size: 1.12em)[Outdoors
      #h(0.3em) $T_"out"$])
    content((0.35, 6.85), anchor: "west", text(fill: outdoor-dark)[follows the weather of the day])
    rect((4.4, 0.45), (13.4, 6.0), radius: 0.2, fill: white, stroke: 1.6pt + wall-line)
    content((4.65, 5.65), anchor: "west", text(fill: wall-line, weight: "bold")[Room])

    // the two heat stores
    node((5.0, 2.45), (8.0, 4.45), soft, air-c, [Air #h(0.2em) $T_a$],
      [$C_a$ = #fmt(cfg.c_air / 1e6, digits: 0) MJ/K], [fast: #sym.tilde.op#fmt(tau-fast) h])
    node((9.8, 2.45), (12.8, 4.45), mass-soft, mass-c, [Walls & floor #h(0.2em) $T_m$],
      [$C_m$ = #fmt(cfg.c_mass / 1e6, digits: 0) MJ/K], [slow: #sym.tilde.op#fmt(tau-slow, digits: 0) h])

    // conductances
    arrow2((0.6, 3.45), (5.0, 3.45))
    conductance((2.55, 3.45), [#int(cfg.ua_air_out) W/K], $U_"ao"$, [windows & air leaks])
    arrow2((8.0, 3.45), (9.8, 3.45))
    conductance((8.9, 3.45), [#int(cfg.ua_air_mass) W/K], $U_"am"$, none)
    arrow2((12.8, 3.45), (15.6, 3.45))
    conductance((14.55, 3.45), [#int(cfg.ua_mass_out) W/K], $U_"mo"$, [outer walls])

    // heat pump
    rect((5.0, 0.8), (8.0, 1.7), radius: 0.14, fill: white, stroke: 1.2pt + ink)
    content((6.5, 1.25), text(weight: "bold")[Heat pump #h(0.2em) $Q_"hvac"$])
    arrow1((6.1, 1.75), (6.1, 2.4), color: heat, thickness: 1.7pt)
    arrow1((6.9, 2.4), (6.9, 1.75), color: cool, thickness: 1.7pt)
    content((5.95, 2.08), anchor: "east", text(fill: heat, weight: "semibold", size: 0.95em)[heat])
    content((7.05, 2.08), anchor: "west", text(fill: cool, weight: "semibold", size: 0.95em)[cool])
    content((8.35, 1.6), anchor: "west", text(size: 1.04em)[$Q_"hvac" = a · #int(cfg.q_hvac_max) "W"$, #h(0.2em)
      $a in [-1, 1]$])
    content((8.35, 1.0), anchor: "west", text(fill: muted, size: 0.95em)[the agent's action, every 15 min; \
      draws $|a| · #fmt(cfg.q_hvac_max / cfg.cop / 1000, digits: 0)$ kW of electricity (COP #int(cfg.cop))])

    // sun
    let (sx, sy) = (8.9, 6.85)
    circle((sx, sy), radius: 0.22, fill: sun-c, stroke: none)
    for i in range(8) {
      let ang = i * 45deg
      line((sx + 0.32 * calc.cos(ang), sy + 0.32 * calc.sin(ang)), (sx + 0.46 * calc.cos(ang), sy + 0.46 * calc.sin(ang)),
        stroke: (paint: sun-c, thickness: 1.3pt, cap: "round"))
    }
    content((sx + 0.65, sy), anchor: "west", text(fill: sun-dark, weight: "bold")[Sun #h(0.2em) $Q_"sun"$
      #text(weight: "regular")[up to #fmt(cfg.q_solar_peak / 1000, digits: 0) kW]])
    bezier((sx - 0.3, sy - 0.45), (7.4, 4.5), (7.95, 5.9), stroke: 1.2pt + sun-c, mark: (end: "stealth", ..tip))
    bezier((sx + 0.3, sy - 0.45), (10.4, 4.5), (9.85, 5.9), stroke: 1.2pt + sun-c, mark: (end: "stealth", ..tip))
    content((8.05, 4.8), anchor: "west", text(fill: sun-dark, weight: "semibold")[#calc.round(100 * cfg.solar_to_air)%])
    content((9.75, 4.8), anchor: "east", text(fill: sun-dark, weight: "semibold")[#calc.round(100 * (1 - cfg.solar_to_air))%])

    // internal gains
    content((4.6, 5.0), anchor: "west", text(fill: internal-c, size: 0.92em)[people & devices #h(0.1em) $Q_"int"$])
    arrow1((5.3, 4.78), (5.5, 4.5), color: internal-c)
  })
}
