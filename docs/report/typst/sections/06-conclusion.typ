#import "../lib.typ": *

= Discussion and conclusion <sec-conclusion>

#para[What the three settings do.]
_Energy use_ depends most on the number of epochs: one pass uses #calc.round(100 * (cfg("epochs_1").energy_kwh / base.energy_kwh - 1))% more than ten.
Energy also rises with $lambda$, and with a narrower clip range.
_Comfort violations_ are lowest wherever learning is slow, because a slow agent has not yet approached the edge of the band.
They rise with the clip range, from #val("clip_0.05", "violation_pct", digits: 1)% to #val("clip_0.5", "violation_pct", digits: 1)% of the occupied steps, and with a smaller $lambda$ (about 4% against #val("baseline", "violation_pct", digits: 1)%).
No setting is better than the baseline in both energy and discomfort.
_Training stability_ is governed by the clip range: 0.5 is the only setting that became unstable, and 0.05 and 0.1 are smoother than the baseline but slower.

#para[Why predictions failed.]
Both major errors come from reading too much into the log of a single configuration.
A low clip fraction does not show that clipping is inactive.
And "progress is set by the number of gradient steps" holds only where gradient steps are scarce.
In both cases the ablation itself was needed to see it, which is perhaps the best argument for running one.
Writing the predictions down first made these errors visible; without them, each result would have looked plausible in hindsight.

#para[In practice.]
For this task the defaults are a sound choice.
We would keep 10 or more epochs, a clip range of 0.1 to 0.3 (0.1 if smooth training matters), and $lambda$ of 0.9 to 0.95, and we would spend any remaining effort on training longer.

#para[Limitations.]
The study uses one simplified room with hand-set parameters and synthetic weather.
The budget of #ksteps(data.total_steps) steps ends before convergence, so we compare learning speed rather than final performance.
Settings were varied one at a time, so interactions (for instance between the clip range and the number of epochs) are not covered.
With 10 seeds and no correction for multiple comparisons, borderline results are weak evidence.
Our explanation of why a large $lambda$ leads to too much pre-heating is untested, and we did not examine why so few clipped samples carry so much weight.
