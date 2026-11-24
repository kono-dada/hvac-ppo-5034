#import "../lib.typ": *

= Introduction <sec-intro>

Heating, ventilation and air conditioning (HVAC) is one of the largest controllable uses of energy: buildings account for nearly 40% of the energy consumed in the United States, and about half of that goes to HVAC @wei2017deep.
An HVAC controller pursues two goals that pull against each other, keeping the occupants comfortable and using little energy, and it does so in a system that reacts slowly: a cold building cannot be made warm at the moment people arrive.

#para[Reinforcement learning for HVAC.]
Model predictive control handles this with an explicit model of the building and can notably reduce energy use, but its practical application is still at an early stage @drgona2020all.
Reinforcement learning (RL) instead learns a controller from interaction.
Wei et al. @wei2017deep trained a deep RL controller that, in EnergyPlus simulations, lowered the energy cost relative to a rule-based strategy while keeping the temperature in range; Wang and Hong @wang2020reinforcement review the field.
Open simulation frameworks such as Sinergym @jimenez2021sinergym and BOPTEST @blum2021boptest have since made controllers comparable.
In a simulated data centre, Biemann et al. @biemann2021experimental found that Soft Actor-Critic reached stable performance with ten times less data than on-policy methods.
Manjavacas et al. @manjavacas2024experimental compared several algorithms, PPO among them, and concluded that the right weighting between comfort and energy "depends entirely on the problem and the environment".
These studies compare algorithms with each other.
This project asks a narrower question: what do the settings of one algorithm do?

#para[PPO and its hyperparameters.]
PPO @schulman2017proximal reuses every batch of experience for several epochs and limits each update by clipping.
Its authors compared clip ranges of 0.1, 0.2 and 0.3 on seven simulated robotics tasks (normalised scores 0.76, 0.82 and 0.70, against −0.39 without clipping).
For the advantage estimator GAE, Schulman et al. @schulman2016high report that $lambda$ between 0.9 and 0.99 usually works best.
Andrychowicz et al. @andrychowicz2021what trained more than 250,000 agents and recommend going over each batch several times, $lambda = 0.9$, and a clipping threshold of 0.25 as a starting point.
Engstrom et al. @engstrom2020implementation caution that clipping does not by itself keep the policy inside a trust region: how far an update moves depends on the optimiser.
Finally, deep RL results vary considerably between random seeds @henderson2018deep, so a comparison needs several seeds and a statistical test @colas2019hitchhiker.

#para[This report.]
We build a small room environment in which the best possible controller can be computed exactly (@sec-env), describe the learning problem and the experimental protocol (@sec-method), write down predictions for every setting _before_ running the ablation (@sec-pred), and then report comfort violations, energy use and training stability for 12 configurations with 10 seeds each, analysing where the predictions failed (@sec-results).
