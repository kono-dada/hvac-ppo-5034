#import "../lib.typ": *

We study how three hyperparameters of Proximal Policy Optimization (PPO), namely the clip range, the GAE parameter $lambda$ and the number of epochs, affect a controller that heats and cools a simulated office room.
The room is a two-node thermal model wrapped as a Gymnasium environment: a heat pump acts on the fast-reacting air, slow walls store heat, and the agent trades electricity against discomfort during occupied hours.
Before running the ablation we wrote down #totals.total testable predictions and the rules for judging them, then trained Stable-Baselines3 PPO in 12 configurations with 10 seeds each.
Of the predictions, #totals.hit held, #totals.partial held in part, #totals.miss failed and #totals.undecided could not be decided with 10 seeds.
The number of epochs matters most: with a single pass over each batch the agent stays comfortable but uses #calc.round(100 * (cfg("epochs_1").energy_kwh / base.energy_kwh - 1))% more energy than with the default of ten.
The clip range governs stability: 0.5 was the only unstable setting, and our prediction that widening the range would change nothing was wrong, because we had misread a low clip fraction as "clipping is inactive".
GAE $lambda$ barely moves the cost between 0.8 and 0.95, yet it shifts the policy from pre-heating too little to pre-heating too much.
Within #ksteps(data.total_steps) training steps, no setting beat a tuned thermostat on average.
