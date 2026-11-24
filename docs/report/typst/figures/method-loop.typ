// One round of PPO training: collect, compute advantages, update; the three ablated knobs highlighted.
#import "style.typ": *
#import "@preview/cetz:0.5.3"

#let m = json("data/method.json")

#let render(width: text-width, title: true) = {
  show: theme
  if title {
    heading-block(width: width, [One round of PPO training],
      sub: [The three settings studied in the ablation are shown in red.])
  }
  let knob(body) = box(fill: heat-soft, radius: 6pt, inset: (x: 5pt, y: 2.5pt), text(fill: heat, weight: "medium", body))
  let k = width / 16.1cm
  cetz.canvas(length: 1cm * k, {
    import cetz.draw: *
    let (w, ch, top) = (4.6, 3.5, 4.8)
    let card(x, num, head, color, body, chip: none) = {
      rect((x + 0.05, top - ch - 0.07), (x + w + 0.05, top - 0.07), radius: 0.2, fill: black.transparentize(93%), stroke: none)
      rect((x, top - ch), (x + w, top), radius: 0.2, fill: white, stroke: 0.8pt + faint)
      rect((x, top - 0.72), (x + w, top), radius: (north: 0.2), fill: color, stroke: none)
      circle((x + 0.42, top - 0.36), radius: 0.21, fill: white, stroke: none)
      content((x + 0.42, top - 0.36), text(fill: color, weight: "bold")[#num])
      content((x + 0.78, top - 0.36), anchor: "west", text(fill: white, weight: "semibold", size: 1.12em, head))
      content((x + w / 2, top - 0.72 - (ch - 0.72) / 2 + if chip != none { 0.3 } else { 0 }),
        box(width: (w - 0.6) * 1cm * k, align(center, par(leading: 0.62em, body))))
      if chip != none { content((x + w / 2, top - ch + 0.38), chip) }
    }
    let arrow(a, b) = line(a, b, stroke: 1.3pt + rgb("#5d6670"), mark: (end: "stealth", fill: rgb("#5d6670"), scale: 0.9))

    card(0, 1, [Collect data], cool)[
      #m.n_envs rooms run in parallel,
      #m.n_steps steps each (15 min per step):
      *#m.per_update steps*, about #m.days days.
      Actions include random noise.
    ]
    card(5.7, 2, [Compute advantages], rgb("#c98a1b"), chip: knob[$lambda$: GAE lambda])[
      The value network scores each state;
      GAE turns the rewards into how much better than expected each step was
      #box[($gamma = #m.gamma$)].
    ]
    card(11.4, 3, [Update the networks], comfort, chip: knob[$epsilon$: clip range #h(0.6em) n_epochs: passes])[
      Shuffle the #m.per_update steps into
      #m.minibatches minibatches of #m.batch_size,
      repeat for *n_epochs* passes,
      clipping each change at $epsilon$.
    ]
    arrow((4.68, top - ch / 2), (5.62, top - ch / 2))
    arrow((10.38, top - ch / 2), (11.32, top - ch / 2))
    // loop back
    let y0 = top - ch - 0.15
    bezier((13.7, y0), (2.3, y0), (11.5, y0 - 1.0), (4.5, y0 - 1.0), stroke: (paint: rgb("#5d6670"), thickness: 1.3pt,
      dash: (4pt, 2.5pt)), mark: (end: "stealth", fill: rgb("#5d6670"), scale: 0.9))
    content((8.0, y0 - 0.95), anchor: "north", text(fill: rgb("#5d6670"))[Discard the old data and collect again with the new
      policy; repeat for about #m.rounds rounds (#calc.round(m.total_steps / 1000) k steps).])
  })
}
