#import "../lib.typ": *

We study how three hyperparameters of Proximal Policy Optimization (PPO), namely the clip range, the GAE parameter $lambda$ and the number of epochs, affect a controller that heats and cools a simulated office room.
The room is a two-node thermal model wrapped as a Gymnasium environment: a heat pump acts on the fast-reacting air, slow walls store heat, and the agent trades electricity against discomfort during occupied hours.
Before running the ablation we wrote down predictions for every setting, together with the rules for judging them, then trained Stable-Baselines3 PPO in 12 configurations with 10 seeds each.
Most predictions held in direction; two were qualitatively wrong, and we analyse those.
The number of epochs matters most: with a single pass over each batch the agent stays comfortable but uses #calc.round(100 * (cfg("epochs_1").energy_kwh / base.energy_kwh - 1))% more energy than with the default of ten.
The clip range governs stability: we predicted that widening it would change nothing, because few samples were clipped, yet 0.5 was the only unstable setting.
Post-hoc gradient diagnostics show that the few clipped samples carry a large share of the gradient, and that a wide range weakens the pull of later epochs back towards the starting policy.
GAE $lambda$ barely moves the cost between 0.8 and 0.95, yet it shifts the policy from pre-heating too little to pre-heating too much.
Within #ksteps(data.total_steps) training steps, no setting beat a tuned thermostat on average.
