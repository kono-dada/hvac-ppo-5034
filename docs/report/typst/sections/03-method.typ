#import "../lib.typ": *

= Method <sec-method>

== The decision process

#para[Observation.]
Seven numbers, each scaled to roughly $[-1, 1]$: the air temperature $T_a$, the wall temperature $T_m$, the outdoor temperature, the solar gain, whether the room is occupied during the coming step, and the time of day as a sine and a cosine.

#para[Action and reward.]
The action is the heat-pump command $a_t in [-1, 1]$ of @sec-env.
The reward is the negative cost of the step,
$
  r_t &= -(e_t + 2 d_t), quad e_t = 0.25 abs(a_t), \
  d_t &= 0.25 o_t max(21 - T_(a,t+1), thick T_(a,t+1) - 24, thick 0),
$ <eq-reward>
with $e_t$ in kWh, $d_t$ in K·h, and $o_t = 1$ if the room is occupied during step $t$ and 0 otherwise.
The rewards of a day add up to $-J$ (@eq-cost), so maximising the return is the same as minimising the cost.

#para[Episode end and discount.]
The day ends at midnight as a true terminal state.
The task is the single day and the agent observes the time, which is the first case of Pardo et al. @pardo2018time.
In a pre-experiment (default PPO, 5 seeds, 500k steps), treating midnight as a truncation (the value of the last state is added to the final reward) made the agent keep conditioning the room "for tomorrow" and raised its cost from 4.54 to 5.19.
The discount is $gamma = 0.99$, under which a reward 19 steps ahead still counts 0.83; discounting does not hide the benefit of pre-heating.

== PPO and the three settings

We use PPO as implemented in Stable-Baselines3 @raffin2021stable, with a policy network and a separate value network $V$ of two hidden layers of 64 tanh units each.
The policy outputs a mean $mu(s)$; during training the action is drawn from $cal(N)(mu(s), sigma^2)$ and clipped to $[-1, 1]$.
The action noise $sigma$ is a single learned parameter that starts at 1.
Evaluation uses $mu(s)$ without noise.
Training repeats three steps.

#para[Step 1: collect.]
Eight copies of the room run 256 steps each, giving a batch of 2048 steps, about 21 days.

#para[Step 2: estimate advantages (GAE $lambda$).]
With the one-step surprise $delta_t = r_t + gamma V(s_(t+1)) - V(s_t)$, the advantage of action $a_t$ is @schulman2016high
$ hat(A)_t = sum_(l >= 0) (gamma lambda)^l delta_(t+l). $ <eq-gae>
$lambda$ sets how far the estimate follows actual outcomes before it trusts the value network.
The weight of a later surprise halves after about 3, 6, 11 and 35 steps for $lambda = 0.8$, 0.9, 0.95 and 0.99; with $lambda = 1$ the estimate is the discounted return minus $V(s_t)$, bootstrapped only where a rollout ends mid-day.
A small $lambda$ gives low variance but relies on $V$ being accurate; a large one is unbiased but noisy.

#para[Step 3: update (clip range $epsilon$, `n_epochs`).]
With the probability ratio $rho_t = pi_"new" (a_t|s_t) \/ pi_"old" (a_t|s_t)$, PPO maximises @schulman2017proximal
$ L = EE_t [min(rho_t hat(A)_t, thick "clip"(rho_t, 1 - epsilon, 1 + epsilon) hat(A)_t)]. $ <eq-clip>
Once an action's probability has changed by more than the factor $1 plus.minus epsilon$ in the favourable direction, it earns no further gain.
The batch is shuffled into 32 minibatches of 64, and `n_epochs` passes are made over them: 320 gradient steps per batch at the default of 10.

All other hyperparameters keep their Stable-Baselines3 defaults: learning rate $3 times 10^(-4)$, value-loss coefficient 0.5, no entropy bonus, gradient-norm limit 0.5, and advantages normalised per minibatch.

== Experimental protocol <sec-protocol>

#para[Design.]
We change one setting at a time around the defaults (bold): clip range $epsilon in {0.05, 0.1, bold(0.2), 0.3, 0.5}$, $lambda in {0.8, 0.9, bold(0.95), 0.99, 1}$ and `n_epochs` $in {1, 3, bold(10), 30}$.
The configuration with all three defaults is the _baseline_.
This gives 12 configurations, each trained with 10 seeds for 204,800 steps (#ab.rounds batches, about 2,100 simulated days); one run takes #calc.round(cfg("epochs_1").train_seconds) to #calc.round(cfg("epochs_30").train_seconds) s on a laptop CPU.
The policy is evaluated without noise on 100 fixed test days, generated from seeds that training never uses, every 8,192 steps and at the end.

#para[A caveat on the budget.]
At 204,800 steps the baseline is past its fast phase but has not converged: in the pre-experiments it reached a cost of 4.54 after 500k steps and 4.23 after 1.5M.
The ablation therefore compares how far each setting gets _on the same amount of data_.
It cannot say whether a slow setting would eventually catch up.

#para[Metrics.]
_Energy use_ is $E$.
_Comfort violations_ are $D$ and the share of occupied steps more than 0.1 K outside the band.
_Training stability_ is measured in three ways: the standard deviation of the final cost over seeds; the _setback_, the largest amount by which an evaluation in the second half of training exceeds the best evaluation before it; and the size of the updates, for which Stable-Baselines3 logs `approx_kl` (how far a batch moved the policy) and `clip_fraction` (the share of samples beyond the clip range).

#para[Deciding.]
Every prediction came with a rule for judging it, fixed in advance.
A run yields one number per metric, so a configuration is a sample of 10.
We compare it with the baseline through the Welch interval of the difference of means, which Colas et al. @colas2019hitchhiker found robust at this sample size, unlike the bootstrap.
"Higher" or "lower" holds if the 95% interval excludes zero.
"Same" requires more than a non-significant difference: the 90% interval must lie within a margin fixed in advance, 0.1 for cost and energy and 0.05 K·h for discomfort.
A predicted range holds if the mean falls inside it, holds _in part_ if the mean falls outside but its 95% interval overlaps the range, and fails if that interval lies entirely outside.
Everything else is _undecided_; with 10 seeds, most predictions of "same" end here.
We apply no correction for multiple comparisons, so a single borderline result should not be read as strong evidence.
In the discussion we use the rules as a check, and we analyse in depth only the results that contradict a prediction _qualitatively_.

#para[Code.]
The environment, the training and evaluation scripts, the analysis and the scripts that generate every number and figure of this report are submitted with it; we used Stable-Baselines3 2.9.2 and Gymnasium 1.4.0.
