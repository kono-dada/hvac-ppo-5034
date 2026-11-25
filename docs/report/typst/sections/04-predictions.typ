#import "../lib.typ": *

= Predictions, made before the experiment <sec-pred>

We wrote down a prediction for every configuration, and the rule for judging it, before any configuration other than the baseline had been trained, and did not change them afterwards.
A prediction stated in advance can turn out wrong, and the places where it does are the results most worth analysing.

#para[Why the literature is not enough.]
The studies of @sec-intro suggest the same shape for all three settings: the default works well and both extremes are worse.
They come from robotics tasks scored by a single return, however, and say neither how a worse setting would show up in our two metrics, energy and discomfort, nor whether a setting matters in this room at all.
A setting can change the result only through the part of PPO it controls, and only if that part is what limits learning here.
Before predicting, we therefore put one general question and one question per setting to the training log of the baseline (5 seeds).

#para[O1. What does slower learning look like? Two phases.]
In the first 25k steps the agent learns to switch on: discomfort falls from 35 K·h to zero, but at a wasteful 7.8 kWh per day.
The remaining 180k steps go into saving energy, down to 4.6 kWh, while discomfort creeps back up to about 0.1 K·h as the policy approaches the edge of the band.
A setting that merely learns more slowly therefore stops earlier on this path, with _more_ energy but _fewer_ violations, not with more of both.

#para[O2. What limits progress, and do epochs control it? The number of gradient steps.]
Saving energy means running close to the edge of the band, which a noisy policy cannot afford (@sec-challenges), so progress in the second phase is tied to the fall of the action noise $sigma$.
$sigma$ is a single parameter, and Adam changes $ln sigma$ by at most about one learning rate per gradient step.
In the baseline $sigma$ fell at about 60% of this limit.
The number of gradient steps, which `n_epochs` sets (32 per pass over a batch), is thus what holds learning back.

#para[O3. Does the clip range bind? Seemingly rarely.]
The clip range acts only on samples whose probability ratio leaves $1 plus.minus epsilon$.
With $epsilon = 0.2$ only 6% of the samples were clipped during the first half of training.
We concluded that the learning rate, not the clip range, limits the size of an update, in line with Engstrom et al. @engstrom2020implementation.

#para[O4. Can the value network relay delayed benefits? Yes.]
A small $lambda$ cuts the advantage short and leaves the rest to $V$ (@eq-gae), so it can learn to pre-heat only if $V$ is accurate.
From 35k steps on, the explained variance of $V$ stayed above 0.95 in every seed.

From these answers we predicted the following; the ranges and directions are listed next to the results in @tab-results.

- *Epochs have the largest effect* (O2). One pass allows too few gradient steps for $sigma$ to fall: even at the limit it cannot get below 0.38, so the agent stays in the wasteful phase, with high energy and almost no discomfort (O1). Three passes fall short as well. Thirty passes learn faster at first but end no better, with larger updates and larger setbacks.
- *The clip range matters only when narrowed* (O3). 0.05 and 0.1 bind far more often and slow learning down. 0.3 and 0.5 loosen a limit that is rarely reached and change nothing. This deliberately contradicts the PPO paper, where 0.3 was worse than 0.2.
- *GAE $lambda$ matters only when raised* (O4). Between 0.8 and 0.95 the cost stays the same, because the value network relays the delayed benefit of pre-heating. 0.99 and 1 are worse, because noisier advantages slow the fall of $sigma$ (O2).

We also recorded where we expected to be wrong: three passes might do better than predicted, since each of their gradient steps sees fresher data; and $epsilon = 0.5$ might go wrong, since our evidence for O3 was collected with clipping switched on.
