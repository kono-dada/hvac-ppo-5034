// Schematic of the two-node RC room model, laid out for one column of the report (about 8.25 cm).
//
// The two heat stores sit side by side inside the room; each exchanges heat with the outdoors
// through the room's shell (air: upwards through the windows, walls: downwards through the
// outer walls), the heat pump sits under the air, and the sun shines in from above.
#import "style.typ": *
#import "@preview/cetz:0.5.3"

#let cfg = json("data/env.json").config

/// Natural width of the drawing; it scales with `width`, the text does not.
#let design-width = 8.25cm

#let render(width: design-width, title: true) = {
  show: theme
  if title {
    heading-block(width: width, [The room as two heat stores],
      sub: [Heat flows along every arrow in proportion to the temperature difference; the agent only controls
        the heat pump.])
  }
  let (tau-fast, tau-slow) = cfg.time_constants_h
  let wall-line = rgb("#5d6670")
  cetz.canvas(length: 1cm * (width / design-width), {
    import cetz.draw: *
    let tip = (fill: auto, scale: 0.75)
    let arrow2(a, b, color: ink-2, thickness: 1pt) = line(a, b, stroke: (paint: color, thickness: thickness),
      mark: (start: "stealth", end: "stealth", ..tip))
    let arrow1(a, b, color: ink-2, thickness: 1pt) = line(a, b, stroke: (paint: color, thickness: thickness),
      mark: (end: "stealth", ..tip))
    let pill(p, value) = content(p, box(fill: white, stroke: 0.6pt + subtle, radius: 4pt, inset: (x: 3.5pt, y: 2.2pt),
      text(weight: "bold", value)))
    let node(a, b, fill, stroke, head, cap, speed) = {
      rect(a, b, radius: 0.14, fill: fill, stroke: 1.1pt + stroke)
      let cx = (a.at(0) + b.at(0)) / 2
      content((cx, b.at(1) - 0.36), text(fill: stroke, weight: "bold", head))
      content((cx, (a.at(1) + b.at(1)) / 2 - 0.02), text(fill: ink-2, cap))
      content((cx, a.at(1) + 0.34), text(fill: stroke, speed))
    }

    // ---- outdoors and the room
    let (W, H) = (8.25, 8.0)
    let (rx0, ry0, rx1, ry1) = (0.2, 1.25, 8.05, 6.75)
    rect((0, 0), (W, H), radius: 0.25, fill: outdoor-c.transparentize(86%), stroke: none)
    rect((rx0, ry0), (rx1, ry1), radius: 0.18, fill: white, stroke: 1.4pt + wall-line)
    content((rx0 + 0.2, ry0 + 0.3), anchor: "west", text(fill: wall-line, weight: "bold")[Room])
    content((0.3, 0.72), anchor: "west", text(fill: outdoor-dark, weight: "bold")[Outdoors #h(0.2em) $T_"out"$])
    content((0.3, 0.32), anchor: "west", text(fill: outdoor-dark)[the weather of the day])

    // ---- the two heat stores
    let (ny0, ny1) = (3.45, 5.05)
    let air = ((0.5, ny0), (3.35, ny1))
    let mass = ((5.1, ny0), (7.75, ny1))
    node(..air, soft, air-c, [Air #h(0.2em) $T_a$],
      [$C_a$ = #fmt(cfg.c_air / 1e6, digits: 0) MJ/K], [fast: $tilde.op$#fmt(tau-fast) h])
    node(..mass, mass-soft, mass-c, [Walls & floor #h(0.2em) $T_m$],
      [$C_m$ = #fmt(cfg.c_mass / 1e6, digits: 0) MJ/K], [slow: $tilde.op$#fmt(tau-slow, digits: 0) h])
    let ymid = (ny0 + ny1) / 2

    // ---- conductances
    // air <-> walls
    // (the gap is too short to put the value on the arrow: symbol above it, value below it)
    arrow2((3.35, ymid), (5.1, ymid))
    content((4.225, ymid + 0.12), anchor: "south", $U_"am"$)
    pill((4.225, ymid - 0.38), [#int(cfg.ua_air_mass) W/K])
    // air <-> outdoors, through the windows (top)
    let xa = 0.95
    arrow2((xa, ny1), (xa, H - 0.15))
    pill((xa, ry1), [#int(cfg.ua_air_out) W/K])
    content((xa + 0.55, ry1 + 0.62), anchor: "west", [$U_"ao"$ #h(0.25em) #text(fill: muted)[windows & air leaks]])
    // walls <-> outdoors, through the outer walls (bottom)
    let xm = 7.3
    arrow2((xm, ny0), (xm, 0.15))
    pill((xm, ry0), [#int(cfg.ua_mass_out) W/K])
    content((xm - 0.55, ry0 - 0.6), anchor: "east", [$U_"mo"$ #h(0.25em) #text(fill: muted)[outer walls]])

    // ---- heat pump, under the air
    let (hy0, hy1) = (1.7, 2.35)
    rect((0.5, hy0), (3.35, hy1), radius: 0.12, fill: white, stroke: 1.1pt + ink)
    content((1.925, (hy0 + hy1) / 2), text(weight: "bold")[Heat pump #h(0.2em) $Q_"hvac"$])
    arrow1((1.65, hy1 + 0.05), (1.65, ny0 - 0.05), color: heat, thickness: 1.5pt)
    arrow1((2.35, ny0 - 0.05), (2.35, hy1 + 0.05), color: cool, thickness: 1.5pt)
    content((1.5, (hy1 + ny0) / 2), anchor: "east", text(fill: heat, weight: "bold")[heat])
    content((2.5, (hy1 + ny0) / 2), anchor: "west", text(fill: cool, weight: "bold")[cool])
    // what the agent sets
    content((3.65, 2.98), anchor: "north-west", block(width: 3.25cm * (width / design-width), {
      set par(leading: 0.5em)
      [#box[$Q_"hvac" = a #sym.dot.op #int(cfg.q_hvac_max)$ W] \
        #text(fill: muted)[#box[$a in [-1, 1]$] is the agent's action, every 15 min; it draws
        #box[$|a| #sym.dot.op #fmt(cfg.q_hvac_max / cfg.cop / 1000, digits: 0)$ kW] of electricity (COP #int(cfg.cop))]]
    }))

    // ---- internal gains
    content((1.25, 5.95), anchor: "west", align(left, text(fill: internal-c)[people & \ devices #h(0.1em) $Q_"int"$]))
    arrow1((1.75, 5.5), (1.95, ny1 + 0.05), color: internal-c, thickness: 0.9pt)

    // ---- sun
    let (sx, sy) = (6.35, 7.35)
    circle((sx, sy), radius: 0.2, fill: sun-c, stroke: none)
    for i in range(8) {
      let ang = i * 45deg
      line((sx + 0.29 * calc.cos(ang), sy + 0.29 * calc.sin(ang)), (sx + 0.42 * calc.cos(ang), sy + 0.42 * calc.sin(ang)),
        stroke: (paint: sun-c, thickness: 1.1pt, cap: "round"))
    }
    content((sx + 0.6, sy + 0.18), anchor: "west", text(fill: sun-dark, weight: "bold")[Sun #h(0.15em) $Q_"sun"$])
    content((sx + 0.6, sy - 0.22), anchor: "west", text(fill: sun-dark)[up to #fmt(cfg.q_solar_peak / 1000, digits: 0) kW])
    // 70% into the walls (straight down), 30% into the air (curving left)
    arrow1((sx, sy - 0.5), (sx, ny1 + 0.05), color: sun-c, thickness: 1.1pt)
    content((sx + 0.12, 5.8), anchor: "west", text(fill: sun-dark, weight: "bold")[#calc.round(100 * (1 - cfg.solar_to_air))%])
    bezier((sx - 0.38, sy - 0.32), (3.2, ny1 + 0.05), (4.3, 7.0), stroke: 1.1pt + sun-c, mark: (end: "stealth", ..tip))
    content((4.1, 6.15), anchor: "east", text(fill: sun-dark, weight: "bold")[#calc.round(100 * cfg.solar_to_air)%])
  })
}
