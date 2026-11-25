#import "../lib.typ": *
#import "../figures/env-schematic.typ" as schematic
#import "../figures/env-controllers.typ" as controllers

= The environment: one room, one heat pump <sec-env>

== The room model

A 30 m#super[2] office has a reversible heat pump.
People are present from 8:00 to 18:00, and during these hours the air should stay between 21 and 24 °C, the _comfort band_ of a BOPTEST test case @arroyo2022comparison.
At other times any temperature is acceptable.
Every 15 minutes the agent decides how hard the heat pump heats or cools.

We describe the room by two temperatures (@fig-room): the air $T_a$, which is what people feel, and the walls and floor $T_m$, which store most of the heat:
$
  C_a (dif T_a) / (dif t) &= U_"ao" (T_"out" - T_a) + U_"am" (T_m - T_a) \
  &quad + med Q_"hvac" + f Q_"sun" + Q_"int", \
  C_m (dif T_m) / (dif t) &= U_"am" (T_a - T_m) + U_"mo" (T_"out" - T_m) \
  &quad + med (1 - f) Q_"sun".
$ <eq-room>
Each line is an energy balance: the left side says how fast a body warms up, the right side adds up the heat flowing into it, and every exchange term is a conductance times a temperature difference.
Only $Q_"hvac"$ is under the agent's control.
The action $a in [-1, 1]$ sets $Q_"hvac" = 3000 a$ W; with a coefficient of performance of 3 the heat pump draws $abs(a)$ kW, at most 0.25 kWh per step.
The heat pump acts on the air alone; the walls only warm up through the air.

The parameters are $C_a = 10^6$ and $C_m = 10^7$ J/K, $U_"ao" = 30$, $U_"am" = 400$ and $U_"mo" = 50$ W/K, and $f = 0.3$; people and equipment give off $Q_"int" = 400$ W when the room is occupied and 50 W otherwise.
They make the air fast and the walls slow: one step at full power moves the air by 2.2 °C, the air responds on a time scale of about 0.6 h, and the room as a whole drifts towards the outdoor temperature on one of about 39 h.
Because the model is linear and the inputs are held constant within a step, each step is computed exactly with a matrix exponential, without integration error.

#column-figure(
  from-module(schematic.render),
  placement: top,
  caption: [The room as two heat stores. Heat flows along every arrow in proportion to the temperature difference (@eq-room); the agent controls only the heat pump.],
  label: <fig-room>,
)

#para[How realistic is it?]
Models made of a few heat capacities and conductances are a standard tool in building control; Bacher and Madsen @bacher2011identifying identify such models from measurements of a real building and find that a single temperature does not describe its dynamics.
Two temperatures are the minimum that makes the air respond quickly _and_ lets the room keep its heat overnight.
Our parameters are set by hand to plausible magnitudes, not fitted to data, and the model ignores humidity, ventilation, neighbouring rooms and the dependence of the heat pump's efficiency on the weather.
Our conclusions hold for this simplified room, not for real buildings.

== Days, cost and reference controllers

An episode is one day of 96 steps, and every episode draws a new day.
The daily mean outdoor temperature is uniform between −2 and 32 °C, so the same policy must heat in winter and cool in summer.
A daily swing of 6 to 12 K, random fluctuations of about 1 K and a sun of random strength are added, and the walls start cold on cold days and warm on warm ones.

Each day yields two numbers: the energy $E$ (kWh of electricity) and the discomfort $D$ (K·h, the convention of BOPTEST @blum2021boptest): degrees outside the comfort band times hours, counted only while the room is occupied.
They are combined into one cost,
$ J = E + 2 D, $ <eq-cost>
so that one K·h of discomfort is as bad as two hours of the heat pump at full power.
A weighted sum of this kind is the usual choice, for example in Sinergym @jimenez2021sinergym.
With the weight 2 the best possible controller has no discomfort at all (@tab-controllers).
Any violation by the agent is thus a control error rather than a deliberate trade, which gives the two metrics distinct meanings: discomfort measures precision, energy measures efficiency.

#column-table(
  table(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    rule-heavy,
    table.header([Controller], [$E$ (kWh)], [$D$ (K·h)], [Viol. (%)], [Cost $J$]),
    rule-light,
    ..data.controllers.map(c => (
      [#c.label],
      num(c.energy_kwh),
      num(c.discomfort_kh),
      num(c.violation_pct, digits: 1),
      if c.key in ("thermostat_tuned", "oracle") { strong(num(c.cost)) } else { num(c.cost) },
    )).flatten(),
    rule-heavy,
  ),
  caption: [Reference controllers on the 100 test days (daily means). Viol.: share of occupied steps more than 0.1 K outside the comfort band.],
  label: <tab-controllers>,
)

@tab-controllers gives the scale.
The _oracle_ knows the whole day's weather in advance; because the model is linear, its optimal actions are the solution of a linear program.
It is a lower bound on the cost, not a fair competitor.
The _tuned thermostat_ starts at 5:30, aims for 22 to 23 °C, and has its three parameters chosen from 378 combinations on 200 separate days.
It is the bar a learned controller should clear.
@fig-controllers shows why starting early matters: on a cold day a thermostat that switches on at 8:00 reaches the comfort band only at 10:30, whereas the oracle starts heating at 5:15 and then holds the air exactly on the lower edge of the band.
Starting at 3:00 instead cuts the discomfort from #num(ctrl("thermostat_8").discomfort_kh) to #num(ctrl("thermostat_3").discomfort_kh) K·h but costs #num(ctrl("thermostat_3").energy_kwh - ctrl("thermostat_8").energy_kwh, digits: 1) kWh more energy; the tuned thermostat is better than both.

== Why this is hard for PPO <sec-challenges>

#para[Delayed consequences.]
On cold days (daily mean below 5 °C) the oracle first switches on at about 5:40 on average and as early as 3:15, but discomfort is only counted from 8:00.
Up to 19 steps lie between the cost of pre-heating and its benefit.
Assigning that benefit to the right actions is what GAE $lambda$ controls.

#para[The optimum sits on the edge of a cliff.]
The oracle keeps the air within 0.05 °C of the edge of the band during 80% of the occupied time.
One degree of safety margin costs about 0.1 kWh per hour, while one degree outside the band costs 2 kWh per hour.
PPO explores by adding Gaussian noise to its actions, and a noise of 0.1 in the action already moves the air by about 0.2 °C.
The agent can approach the edge only as fast as it reduces its own noise.

#para[Two phases with very different signals.]
Leaving the heat pump off costs #num(ctrl("off").cost, digits: 1), all of it discomfort; the oracle's cost of #num(ctrl("oracle").cost) is all energy.
The agent first learns to switch on, and then has to shave energy with a learning signal an order of magnitude weaker.

#para[No forecast.]
The agent observes the current weather only, so part of the gap to the oracle cannot be closed; we did not measure how large that part is.

#wide-figure(
  from-module(controllers.render),
  caption: [A cold and a hot test day under three controllers. The oracle pre-heats before the room is occupied and then keeps the air exactly on the edge of the comfort band; a thermostat that only switches on at 8:00 is too late on the cold day.],
  label: <fig-controllers>,
)
