#import "../lib.typ": *
#import "../figures/ablation-learning-curves.typ" as curves
#import "../figures/ablation-tradeoff.typ" as tradeoff

// ---------------------------------------------------------------- Table: predictions and results
#let n-cols = 12
#let dash = text(fill: muted)[—]
#let mark-slot = 1.0em  // room after a value for the mark of its prediction

/// The prediction about one headline metric, and the mark saying how it fared.
#let predicted(r, key) = if key in r.predictions { text(fill: muted, r.predictions.at(key).label) } else { dash }
#let fared(r, key) = box(width: mark-slot, align(right, if key in r.predictions { mark(r.predictions.at(key).verdict) }))
/// Bold if the difference from the baseline is significant (95% Welch interval excludes zero).
#let emphasised(r, key, body) = if r.diff.at(key).significant { strong(body) } else { body }
/// Difference with its interval, signed only where needed to keep the column narrow.
#let delta(d) = [#num(d.d, sign: true) #h(0.1em) \[#num(d.lo), #num(d.hi)\]]

#let result-row(r) = (
  [#h(0.7em)#setting(r.value)],
  predicted(r, "cost"),
  [#num(r.cost) ± #num(r.cost_sd)#fared(r, "cost")],
  emphasised(r, "cost", delta(r.diff.cost)),
  predicted(r, "energy_kwh"),
  [#emphasised(r, "energy_kwh", num(r.energy_kwh))#fared(r, "energy_kwh")],
  predicted(r, "discomfort_kh"),
  [#emphasised(r, "discomfort_kh", num(r.discomfort_kh, digits: 3))#fared(r, "discomfort_kh")],
  num(r.violation_pct, digits: 1),
  emphasised(r, "rollback", num(r.rollback)),
  // With 1 or 3 epochs approx_kl covers only the last pass over the batch and is too small.
  if r.name in ("epochs_1", "epochs_3") [(#num(r.approx_kl, digits: 4))] else { num(r.approx_kl, digits: 4) },
  scorecard(r.verdicts),
)

#let group-titles = (
  epochs: [Number of epochs, `n_epochs` (default 10)],
  clip: [Clip range $epsilon$ (default 0.2)],
  lambda: [GAE $lambda$ (default 0.95)],
)

#let results-table = {
  set text(size: 7.3pt)
  let thin = 0.4pt + muted
  table(
    columns: (auto,) * (n-cols - 1) + (1fr,),
    align: (left,) + (center,) * (n-cols - 1),
    inset: (x: 2.5pt, y: 2.6pt),
    rule-heavy,
    table.header(
      table.cell(rowspan: 2, align: left + bottom)[Setting],
      table.cell(colspan: 3)[Cost $J$],
      table.cell(colspan: 2)[Energy $E$ (kWh)],
      table.cell(colspan: 2)[Discomfort $D$ (K·h)],
      table.cell(rowspan: 2, align: bottom)[Viol. \ (%)],
      table.cell(colspan: 2)[Stability],
      table.cell(rowspan: 2, align: bottom)[Predictions \ judged],
      table.hline(y: 1, start: 1, end: 4, stroke: thin),
      table.hline(y: 1, start: 4, end: 6, stroke: thin),
      table.hline(y: 1, start: 6, end: 8, stroke: thin),
      table.hline(y: 1, start: 9, end: 11, stroke: thin),
      [predicted], [mean ± SD], [Δ \[95% CI\]],
      [predicted], [mean],
      [predicted], [mean],
      [setback], [`approx_kl`],
    ),
    rule-light,
    [Baseline],
    dash,
    [#num(base.cost) ± #num(base.cost_sd)#box(width: mark-slot)],
    dash,
    dash,
    [#num(base.energy_kwh)#box(width: mark-slot)],
    dash,
    [#num(base.discomfort_kh, digits: 3)#box(width: mark-slot)],
    num(base.violation_pct, digits: 1),
    num(base.rollback),
    num(base.approx_kl, digits: 4),
    dash,
    ..groups.map(g => (
      rule-faint,
      table.cell(colspan: n-cols, align: left, inset: (top: 4pt, bottom: 2pt), text(weight: "medium", group-titles.at(g.key))),
      ..g.rows.map(result-row).flatten(),
    )).flatten(),
    rule-heavy,
  )
}

// ---------------------------------------------------------------- Table: GAE lambda by time of day
#let lambda-table = table(
  columns: (1fr, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right),
  rule-heavy,
  table.header(
    table.cell(rowspan: 2, align: left + bottom)[],
    table.cell(colspan: 3, align: center)[Energy (kWh)],
    table.cell(colspan: 2, align: center)[Discomfort (K·h)],
    table.hline(y: 1, start: 1, end: 4, stroke: 0.4pt + muted),
    table.hline(y: 1, start: 4, end: 6, stroke: 0.4pt + muted),
    [before 8], [8–18], [after 18], [8–10], [10–18],
  ),
  rule-light,
  ..data.lambda_by_time.map(r => {
    let name = if r.kind == "reference" [#r.label.split(",").first()] else if r.kind == "baseline" [$lambda = #r.label$ (baseline)] else [$lambda = #r.label$]
    (
      (if r.label == "0.8" { (rule-faint,) } else { () })
        + (name, strong(num(r.energy_before)), num(r.energy_occupied), num(r.energy_after),
          num(r.discomfort_first2h, digits: 3), num(r.discomfort_rest, digits: 3))
    )
  }).flatten(),
  rule-heavy,
)

// ---------------------------------------------------------------- numbers used in the text
#let all-rows = groups.map(g => g.rows).flatten()
#let clip-rows = groups.find(g => g.key == "clip").rows
/// How far an update moves the policy relative to the clip range: sqrt(2 KL) / epsilon.
#let reach = (clip-rows.map(r => (r.value, r.approx_kl)) + ((0.2, base.approx_kl),)).map(((eps, kl)) => calc.sqrt(2 * kl) / eps)
/// Spread over seeds relative to the baseline, for the settings where it is not significantly different.
#let quiet-sd = all-rows.filter(r => not r.sd_ratio.significant).map(r => r.sd_ratio.ratio)
#let pct(name) = num(100 * cfg(name).clip_fraction, digits: 1) + "%"

= Results <sec-results>

All #ab.n_runs runs finished.
@tab-results sets every prediction next to its result; @fig-curves shows the learning curves.

#wide-table(
  results-table,
  caption: [Predictions (grey, fixed beforehand) and results after #ksteps(data.total_steps) training steps: daily means over 100 test days, 10 seeds. Marks show how each prediction fared (#mark-legend). The last column counts all judged predictions of a setting, including those about stability. Δ: difference from the baseline with its 95% Welch interval; bold: the interval excludes zero. Viol.: share of occupied steps outside the comfort band. Setback: largest rise of the evaluation cost above its earlier best in the second half of training. `approx_kl` is measured during the last pass over each batch only; with 1 or 3 passes (in parentheses) it is not comparable with the other rows.],
  label: <tab-results>,
)

== Overview

#para[Scorecard.]
Of the #totals.total predictions, #totals.hit held, #totals.partial held in part, #totals.miss failed and #totals.undecided remained undecided.
Most of the undecided ones claimed "same as the baseline", which 10 seeds are too few to confirm within our margins.
The full list is in the repository (`docs/report/data/ablation_verdicts.md`).

#para[PPO did not beat the thermostat.]
After #ksteps(data.total_steps) steps no configuration is better on average than the tuned thermostat (#num(ctrl("thermostat_tuned").cost)); only #ab.runs_below_thermostat.at("epochs_30") of the #ab.n_runs runs are, both with 30 epochs.
This is the budget caveat of @sec-protocol: the agents are still learning.

#para[The baseline is noisier than we thought.]
The five seeds behind the predictions gave #num(ab.baseline_seeds.first5_mean) ± #num(ab.baseline_seeds.first5_sd). All ten give #val("baseline", "cost") ± #val("baseline", "cost_sd"), and a single seed (#num(ab.baseline_seeds.worst_cost)) accounts for almost all of the additional spread.

#para[Most settings learn the same thing at different speeds.]
In @fig-tradeoff the black line is the path of the baseline configuration during training: it starts wasteful but comfortable and moves towards less energy and slightly more discomfort.
Most settings end on or near this path.
Those that only learn more slowly (1 and 3 epochs, $epsilon = 0.05$, $lambda = 0.99$ and 1) sit where the baseline was earlier, as O1 predicted: $epsilon = 0.05$ ends where the baseline was after #ksteps(cfg("clip_0.05").baseline_steps_to_reach) steps, and three epochs where it was after #ksteps(cfg("epochs_3").baseline_steps_to_reach).
Thirty epochs and $lambda = 0.8$ and 0.9 lie further along, saving energy at the price of more violations.
Only $epsilon = 0.5$ leaves the path.

#wide-figure(
  from-module(curves.render),
  caption: [Learning curves: cost, energy and discomfort on the 100 test days during training (mean of 10 seeds; band: ±1 standard deviation). Values below the baseline are blue, values above it orange to red.],
  label: <fig-curves>,
)

== Number of epochs: right direction, wrong conversion

One epoch behaved as predicted.
Its cost of #val("epochs_1", "cost") is all energy, its $sigma$ is still #val("epochs_1", "sigma_end") at the end, and it is stuck where the baseline was after #ksteps(cfg("epochs_1").baseline_steps_to_reach) steps, having learned to switch on but not to save.

Three epochs did better than predicted (#val("epochs_3", "cost"), just below the range).
We had converted gradient steps into progress using the baseline's own curve, but the baseline needed about 13,400 gradient steps to get where three epochs got with 9,600.
O2 itself holds: measured against the bound of one learning rate per gradient step, $ln sigma$ fell at #val("epochs_1", "sigma_drop_vs_bound") and #val("epochs_3", "sigma_drop_vs_bound") of the maximal rate with 1 and 3 epochs, but only at #val("baseline", "sigma_drop_vs_bound") and #val("epochs_30", "sigma_drop_vs_bound") with 10 and 30.
The conversion failed because the baseline spends its early gradient steps on learning to switch on, not on reducing $sigma$.

Thirty epochs learn faster early on (cost #num(cfg("epochs_30").cost_at.at("49152")) against #num(base.cost_at.at("49152")) after 49k steps), but the lead shrinks to an insignificant #num(cfg("epochs_30").diff.cost.d, sign: true) at the end.
They buy slightly lower energy with updates #num(cfg("epochs_30").approx_kl / base.approx_kl, digits: 1) times as large, setbacks of #val("epochs_30", "rollback") instead of #val("baseline", "rollback"), and #num(cfg("epochs_30").train_seconds / base.train_seconds, digits: 1) times the computing time.

== Clip range: clipping was active after all

This is where we were most wrong.
If the learning rate alone limited the size of an update, `approx_kl` would not depend on $epsilon$.
In fact it grows #calc.round(cfg("clip_0.5").approx_kl / cfg("clip_0.05").approx_kl)-fold from $epsilon = 0.05$ to 0.5.
A typical change of the probability ratio is $sqrt(2 "KL")$, and for all five values it lies between #num(calc.min(..reach), digits: 1) and #num(calc.max(..reach), digits: 1) times $epsilon$: the size of an update scales with the clip range.
The clip range, not the learning rate, limits the update, and O3 was a misreading.
`clip_fraction` counts how many samples lie beyond the boundary, not how much they matter.
It stays small throughout, and even _falls_ from #pct("clip_0.05") at $epsilon = 0.05$ to #pct("clip_0.5") at 0.5 while the updates grow.
The log of a single setting cannot show how strongly those few samples hold an update back.
Only varying $epsilon$ can.

A clip range of 0.05 is slower than the baseline (#diff("clip_0.05")), although less so than predicted, and it is the smoothest setting.
Ranges of 0.1 and 0.3 cannot be told apart from the baseline in cost, but 0.3 already doubles the setback.
A range of 0.5 is unstable, as we had feared: its cost is higher (#diff("clip_0.5")), its spread over seeds is #num(cfg("clip_0.5").sd_ratio.ratio, digits: 1) times that of the baseline, and its setback is #val("clip_0.5", "rollback").
It learns faster than the baseline at first and keeps pace until about 150k steps; then it stops improving and starts to oscillate (@fig-curves).
By then the policy runs close to the edge of the band, where one large update is enough to push the room outside.
Single updates reach an `approx_kl` of #num(cfg("clip_0.5").approx_kl_max, digits: 1), far beyond the 0.19 that is possible if every ratio stays within $1 plus.minus 0.5$: clipping does not confine the ratio, as Engstrom et al. @engstrom2020implementation observed.
Compared with the literature, 0.1 to 0.3 are equivalent here, and "both ends are worse" holds only for 0.05 and 0.5.

#wide-figure(
  from-module(tradeoff.render),
  caption: [Energy against discomfort. The black line is the path of the baseline configuration during training; the markers show where each setting ends up (mean of 10 seeds, with 95% confidence intervals in panel b). Dotted lines connect points of equal cost.],
  label: <fig-tradeoff>,
)

== GAE $lambda$: same cost, different policy

All five predictions about the cost held: 0.8 and 0.9 match the baseline, while 0.99 and 1 are worse (#diff("lambda_0.99") and #diff("lambda_1")).
The prediction that energy and discomfort would also stay the same did not.
From 0.8 to 1, $lambda$ moves the policy steadily from "cheap but late" to "safe but wasteful".
@tab-lambda shows where: the larger $lambda$, the more energy is spent on pre-heating before 8:00.
With $lambda = 0.8$ the agent pre-heats too little and pays for it in the first two occupied hours; its saving in energy and its additional discomfort cancel in the cost.
The value network does relay the delayed benefit of pre-heating, as we had argued, but not all of it.
With $lambda = 0.99$ and 1 the agent pre-heats about three times as much as the oracle.
Their $sigma$ also falls more slowly, as predicted, but we had not foreseen that the damage would take the form of too much pre-heating.
A possible explanation, which we have not tested, is that a large $lambda$ credits the morning's comfort directly to the pre-heating actions, so the policy learns quickly that pre-heating pays, but not when to stop.
All PPO policies also still spend energy after 18:00, which is pure waste.

#column-table(
  lambda-table,
  caption: [Where the energy goes and where the discomfort arises, by time of day: the five values of $lambda$, the oracle and the tuned thermostat (daily means over the 100 test days; PPO: 10 seeds).],
  label: <tab-lambda>,
)

== Training stability

#para[Spread over seeds.]
Only one epoch and $epsilon = 0.5$ have a significantly larger spread than the baseline, both about 3.9 times as large.
For the other nine settings the ratio lies between #num(calc.min(..quiet-sd)) and #num(calc.max(..quiet-sd)), which 10 seeds cannot resolve.

#para[Setbacks]
follow the size of the updates: they are smallest for $epsilon = 0.05$ and 0.1, intermediate for $epsilon = 0.3$ and 30 epochs, and largest for $epsilon = 0.5$.
One epoch is the exception, and it exposes a weakness of the metric.
Its curve is flat in the second half, so every wobble counts as a setback, although nothing that had been learned is lost.
Its setbacks mean "not there yet"; those of $epsilon = 0.5$ mean "broke what it had".
