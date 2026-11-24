#import "../lib.typ": *

= Predictions, made before the experiment <sec-pred>

We wrote the predictions down and committed them to the repository before any configuration other than the baseline had been trained (commits `8a7a3a4` and `ab7f73b`; the results follow in `42917a8`).
They rest on the literature of @sec-intro and on the training log of the baseline with 5 seeds, in which we saw three things.

#para[O1. Learning has two phases.]
In the first 25k steps the agent learns to switch on: discomfort falls from 35 K·h to zero, but at a wasteful 7.8 kWh per day.
The remaining 180k steps go into saving energy, down to 4.6 kWh, while discomfort creeps back up to about 0.1 K·h as the policy approaches the edge of the band.
It follows that a setting which merely learns more slowly stops earlier on this path, with _more_ energy but _fewer_ violations.

#para[O2. Saving energy is tied to the action noise.]
A noisy policy has to keep its distance from the edge.
The noise $sigma$ is one parameter, and Adam changes $ln sigma$ by at most about one learning rate per gradient step, so the number of gradient steps limits how fast $sigma$ can fall.

#para[O3. Clipping seemed to be rarely active.]
With $epsilon = 0.2$ only 6% of the samples were clipped during the first half of training.
We concluded that the learning rate, not the clip range, limits the size of an update, in line with Engstrom et al. @engstrom2020implementation.

From these observations we predicted the following; the ranges and directions are listed next to the results in @tab-results.

- *Epochs have the largest effect.* One pass allows too few gradient steps for $sigma$ to fall (it cannot get below 0.38), so the agent stays in the wasteful phase; three passes fall short as well. Thirty passes learn faster at first but end no better, with larger updates and larger setbacks.
- *The clip range matters only when narrowed.* 0.05 and 0.1 slow learning down. 0.3 and 0.5 change nothing, because of O3. This deliberately contradicts the PPO paper, where 0.3 was worse than 0.2.
- *GAE $lambda$ matters only when raised.* Between 0.8 and 0.95 the cost stays the same, because the value network relays the delayed benefit of pre-heating. 0.99 and 1 are worse, because noisier advantages slow the fall of $sigma$.

We also recorded where we expected to be wrong: three passes might do better than predicted, since each of their gradient steps sees fresher data; and $epsilon = 0.5$ might go wrong, since our evidence for O3 was collected with clipping switched on.
