"""PPO with extra, read-only diagnostics of what clipping does inside each update.

`DiagnosticPPO.train()` is a copy of `stable_baselines3.PPO.train()` (version
2.9.2) with measurements added around it. The measurements never consume random
numbers, never write to `.grad`, and never touch parameters or optimizer state,
so a run with the same seed is bit-for-bit the same as with plain PPO (checked
in tests/test_diagnostics.py and by scripts/analyze_clip_diagnostics.py).

What is recorded, per epoch (one pass over the 2048 samples of a round):

  Which samples are past the clip boundary. With A the (minibatch-normalised)
  advantage, r the probability ratio and eps the clip range, let
  u = sign(A) (r - 1) / eps, i.e. how far the sample has moved in the direction
  the advantage favours, in units of eps.
    u > 1           "zeroed": the min() picks the clipped term, gradient exactly 0
    u < -1          past the boundary against the advantage: NOT clipped, the
                    gradient keeps pulling it back
    |r - 1| > eps   SB3's clip_fraction counts both of the above.
  Additionally separate float32 boundary roundoff and equal-value surrogate ties
  (min() assigns half the gradient to each branch), rather than incorrectly
  zeroing the gradient at clamp's endpoints.

  Counterfactual policy gradients on each minibatch (actor parameters only):
    g_u  gradient of the unclipped surrogate  -mean(A r)
    g_z  the part of g_u that comes from zeroed samples
    g_c  gradient actually used (read from .grad after backward); g_c = g_u - g_z

  Consistency of the 32 minibatch gradients of an epoch, for g_c and g_u:
    ||mean_k g_k||^2 / mean_k ||g_k||^2  (1 = all identical; about 1/32 for pure noise)

  Gradient-norm clipping (max_grad_norm, applied to actor and critic together):
  the total norm before clipping, the actor and critic parts, whether it fired,
  and whether it would have fired with g_u in place of g_c.

  Parameter movement of the actor: length of each optimizer step, cosine
  between consecutive steps.

  At the end of the epoch, on all 2048 samples: exact KL(old || new) between the
  Gaussian policies, SB3's approx_kl estimator, and the distribution of u.

Per round: path length vs net displacement of the actor parameters, and the
shift of the action mean (in units of the old sigma and in action units).

Across rounds (does a round undo what the previous ones did?): the cosine
between the net parameter displacements of consecutive rounds, and over the
last 10 rounds the net change divided by the sum of the per-round changes, both
for the actor parameters and for the action mean on the current observations.
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import stable_baselines3
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

SB3_VERSION = "2.9.2"
# Edges for the distribution of u at the end of each epoch.
U_BINS = (-np.inf, -1.0, 0.0, 0.5, 0.8, 1.0, np.inf)
U_BIN_NAMES = ("u_lt_m1", "u_m1_0", "u_0_05", "u_05_08", "u_08_1", "u_gt_1")
LOOKBACK = 10  # rounds, for the across-round measures


def _flat(tensors) -> th.Tensor:
    return th.cat([t.reshape(-1) for t in tensors])


def _clip_gradient_parts(advantages: th.Tensor, ratio: th.Tensor, eps: float) -> dict[str, th.Tensor]:
    """Match installed Torch's actual clamp/min autograd branches at float32 ties.

    At an exact clamp endpoint, subtraction-based SB3 clip_fraction and u>1
    need not agree with the branch derivative; min may also split a tie's
    gradient. Measure the removed fraction on a detached scalar subgraph.
    """
    with th.enable_grad():
        r = ratio.detach().requires_grad_()
        a = advantages.detach()
        surrogate = a * r
        clipped = a * th.clamp(r, 1 - eps, 1 + eps)
        derivative = th.autograd.grad(th.min(surrogate, clipped).sum(), r)[0]
    with th.no_grad():
        keep = th.where(a != 0, derivative / a, th.ones_like(a))
        weight = 1 - keep
        sb3_outside = th.abs(ratio - 1) > eps
        partial = (weight > 0) & (weight < 1)
        at_endpoint = (ratio == 1 - eps) | (ratio == 1 + eps)
        return {
            "weight": weight,
            "zeroed": (sb3_outside & (weight == 1)).float(),
            "against": (sb3_outside & (weight == 0)).float(),
            "boundary_roundoff": (sb3_outside & partial & at_endpoint).float(),
            "surrogate_tie": (sb3_outside & partial & ~at_endpoint).float(),
            "partial_not_sb3": (~sb3_outside & partial).float(),
        }


def _consistency(vectors: list[th.Tensor]) -> tuple[float, float]:
    """||mean||^2 / mean ||g||^2, and the mean cosine of each vector to the mean."""
    g = th.stack(vectors)
    m = g.mean(0)
    ratio = (m.square().sum() / g.square().sum(1).mean()).item()
    cos = (g @ m / (g.norm(dim=1) * m.norm() + 1e-30)).mean().item()
    return ratio, cos


class DiagnosticPPO(PPO):
    def __init__(self, *args, **kwargs):
        if stable_baselines3.__version__ != SB3_VERSION:
            raise RuntimeError(
                f"DiagnosticPPO copies PPO.train() from SB3 {SB3_VERSION}, found {stable_baselines3.__version__}"
            )
        super().__init__(*args, **kwargs)
        self.diag_epochs: list[dict] = []
        self.diag_rounds: list[dict] = []
        # one entry per recent round: actor parameters and a copy of the mean network at the
        # start of the round, the round's net displacement and action-mean shift
        self._history: deque = deque(maxlen=LOOKBACK)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["diag_epochs", "diag_rounds", "_history"]

    def _mean_net(self) -> th.nn.Module:
        """Observation -> action mean, for the MLP policy with a flatten feature extractor."""
        return th.nn.Sequential(th.nn.Flatten(), self.policy.mlp_extractor.policy_net, self.policy.action_net)

    def _actor_critic_params(self) -> tuple[list[th.nn.Parameter], list[th.nn.Parameter]]:
        actor, critic = [], []
        for name, p in self.policy.named_parameters():
            is_actor = name == "log_std" or name.startswith(("mlp_extractor.policy_net", "action_net"))
            (actor if is_actor else critic).append(p)
        assert self.policy.log_std is actor[0], "log_std must be the first actor parameter (logstd_g_* columns)"
        return actor, critic

    def _prepare_gradient_probe(self, **context):
        return None

    def _before_gradient_probe_step(self, probe, actor_params, critic_params, g_c):
        pass

    def _after_gradient_probe_step(self, probe, actor_params):
        pass

    def _full_batch(self) -> dict[str, th.Tensor]:
        """All samples of the round, flattened like RolloutBuffer.get() does, without shuffling."""
        buf = self.rollout_buffer
        out = {}
        for key in ("observations", "actions", "log_probs", "advantages"):
            arr = buf.__dict__[key]
            if not buf.generator_ready:
                arr = BaseBuffer.swap_and_flatten(arr)
            out[key] = buf.to_torch(arr.astype(np.float32, copy=False))
        out["log_probs"] = out["log_probs"].flatten()
        out["advantages"] = out["advantages"].flatten()
        return out

    @th.no_grad()
    def _mean_and_log_std(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        dist = self.policy.get_distribution(obs).distribution
        return dist.mean.flatten(), th.log(dist.stddev).flatten()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # These diagnostics are specific to the project's deterministic-forward,
        # separate-network 1-D Gaussian MLP policy. Fail instead of silently
        # treating entropy/feature-network gradients as surrogate gradients.
        from stable_baselines3.common.torch_layers import FlattenExtractor

        if (not isinstance(self.action_space, spaces.Box) or self.action_space.shape != (1,)
                or self.ent_coef != 0 or self.use_sde
                or not isinstance(self.policy.features_extractor, FlattenExtractor)
                or any(isinstance(m, (th.nn.Dropout, th.nn.modules.batchnorm._BatchNorm))
                       for m in self.policy.modules())
                or any(p.requires_grad for p in self.policy.features_extractor.parameters())):
            raise ValueError("DiagnosticPPO requires a 1-D Gaussian flatten-MLP policy, ent_coef=0, and no SDE/dropout/batchnorm.")
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # ---- diagnostics: state at the start of the round
        actor_params, critic_params = self._actor_critic_params()
        full = self._full_batch()
        mu_old, logstd_old = self._mean_and_log_std(full["observations"])
        adv_full = full["advantages"]
        adv_full = (adv_full - adv_full.mean()) / (adv_full.std() + 1e-8)
        theta_start = _flat(actor_params).detach().clone()
        mean_net_start = copy.deepcopy(self._mean_net())
        round_index = len(self.diag_rounds) + 1
        path_length = 0.0
        prev_step = None

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            diag_mb: dict[str, list[float]] = {}
            gc_list: list[th.Tensor] = []
            gu_list: list[th.Tensor] = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # ---- diagnostics: counterfactual gradients (do not touch .grad)
                with th.no_grad():
                    parts = _clip_gradient_parts(advantages, ratio, clip_range)
                    zeroed = parts["zeroed"]
                n = len(advantages)
                g_u = _flat(th.autograd.grad(-policy_loss_1.mean(), actor_params, retain_graph=True))
                g_z = _flat(th.autograd.grad(-(policy_loss_1 * parts["weight"]).sum() / n, actor_params, retain_graph=True))
                probe = self._prepare_gradient_probe(
                    advantages=advantages, ratio=ratio, actor_params=actor_params,
                    observations=rollout_data.observations, mean_net_start=mean_net_start,
                    round=round_index, epoch=epoch + 1,
                    minibatch=len(gc_list) + 1, clip_range=clip_range,
                )

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # ---- diagnostics: gradients actually used, before clipping
                with th.no_grad():
                    g_c = _flat([p.grad for p in actor_params]).clone()
                    critic_norm = _flat([p.grad for p in critic_params]).norm().item()
                    theta_before = _flat(actor_params).clone()

                self._before_gradient_probe_step(probe, actor_params, critic_params, g_c)
                # Clip grad norm
                total_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm).item()
                self.policy.optimizer.step()
                self._after_gradient_probe_step(probe, actor_params)

                # ---- diagnostics: record
                with th.no_grad():
                    step = _flat(actor_params) - theta_before
                    step_norm = step.norm().item()
                    path_length += step_norm
                    step_cos = (
                        (step @ prev_step / (step_norm * prev_step.norm() + 1e-30)).item() if prev_step is not None else np.nan
                    )
                    prev_step = step
                    gu_norm = g_u.norm().item()
                    total_norm_u = float(np.hypot(gu_norm, critic_norm))
                    rec = {
                        "clip_fraction": clip_fraction,
                        "zeroed_frac": zeroed.mean().item(),
                        "against_frac": parts["against"].mean().item(),
                        "boundary_roundoff_frac": parts["boundary_roundoff"].mean().item(),
                        "surrogate_tie_frac": parts["surrogate_tie"].mean().item(),
                        "partial_not_sb3_frac": parts["partial_not_sb3"].mean().item(),
                        "z_norm_share": g_z.norm().item() / (gu_norm + 1e-30),
                        "z_proj_share": (g_z @ g_u).item() / (gu_norm**2 + 1e-30),
                        "cos_c_u": (g_c @ g_u).item() / (g_c.norm().item() * gu_norm + 1e-30),
                        # log_std is the first actor parameter
                        "logstd_g_u": g_u[0].item(),
                        "logstd_g_z": g_z[0].item(),
                        "logstd_g_c": g_c[0].item(),
                        "identity_residual": (g_c - (g_u - g_z)).norm().item() / (gu_norm + 1e-30),
                        "grad_norm_total": total_norm,
                        "grad_norm_actor": g_c.norm().item(),
                        "grad_norm_critic": critic_norm,
                        "grad_clipped": float(total_norm > self.max_grad_norm),
                        "grad_scale": min(1.0, self.max_grad_norm / (total_norm + 1e-6)),
                        "grad_norm_total_unclipped_surrogate": total_norm_u,
                        "grad_clipped_unclipped_surrogate": float(total_norm_u > self.max_grad_norm),
                        "actor_step": step_norm,
                        "actor_step_cos_prev": step_cos,
                    }
                    for key, value in rec.items():
                        diag_mb.setdefault(key, []).append(value)
                    gc_list.append(g_c)
                    gu_list.append(g_u)

            self._n_updates += 1

            # ---- diagnostics: end of epoch, on all samples of the round
            with th.no_grad():
                mu, logstd = self._mean_and_log_std(full["observations"])
                var_old, var_new = th.exp(2 * logstd_old), th.exp(2 * logstd)
                kl_exact = (logstd - logstd_old + (var_old + (mu_old - mu) ** 2) / (2 * var_new) - 0.5).mean().item()
                _, log_prob_full, _ = self.policy.evaluate_actions(full["observations"], full["actions"])
                log_ratio_full = log_prob_full - full["log_probs"]
                ratio_full = th.exp(log_ratio_full)
                kl_k3_full = th.mean((ratio_full - 1) - log_ratio_full).item()
                u_full = (th.sign(adv_full) * (ratio_full - 1) / clip_range).cpu().numpy()
                hist = np.histogram(u_full, bins=np.array(U_BINS))[0] / len(u_full)
                cons_c, cos_mean_c = _consistency(gc_list) if gc_list else (np.nan, np.nan)
                cons_u, cos_mean_u = _consistency(gu_list) if gu_list else (np.nan, np.nan)
                row = {
                    "round": round_index,
                    "epoch": epoch + 1,
                    "clip_range": float(clip_range),
                    "sigma_start": float(th.exp(logstd_old).mean().item()),
                    "kl_exact": kl_exact,
                    "kl_k3_full": kl_k3_full,
                    "sb3_approx_kl_epoch": float(np.mean(approx_kl_divs)),
                    "mean_shift_rms": th.sqrt((((mu - mu_old) / th.exp(logstd_old)) ** 2).mean()).item(),
                    "consistency_c": cons_c,
                    "consistency_u": cons_u,
                    "cos_to_mean_c": cos_mean_c,
                    "cos_to_mean_u": cos_mean_u,
                    **{name: float(v) for name, v in zip(U_BIN_NAMES, hist)},
                    **{key: float(np.nanmean(vals)) for key, vals in diag_mb.items()},
                    "identity_residual_max": float(np.max(diag_mb.get("identity_residual", [np.nan]))),
                }
                self.diag_epochs.append(row)

            if not continue_training:
                break

        # ---- diagnostics: per round
        with th.no_grad():
            theta_end = _flat(actor_params)
            displacement = theta_end - theta_start
            net = displacement.norm().item()
            last = self.diag_epochs[-1]
            mu_end, _ = self._mean_and_log_std(full["observations"])
            mu_shift_abs = th.sqrt(((mu_end - mu_old) ** 2).mean()).item()
            prev = self._history[-1] if self._history else None
            self._history.append(
                {"theta_start": theta_start, "mean_net_start": mean_net_start, "net": net,
                 "displacement": displacement, "mu_shift_abs": mu_shift_abs}
            )
            across = {
                "cos_prev_round_displacement": (
                    (displacement @ prev["displacement"]).item() / (net * prev["net"] + 1e-30) if prev else np.nan
                ),
                "param_efficiency_10": np.nan,
                "mu_efficiency_10": np.nan,
            }
            if len(self._history) == LOOKBACK:
                oldest = self._history[0]
                across["param_efficiency_10"] = (theta_end - oldest["theta_start"]).norm().item() / (
                    sum(h["net"] for h in self._history) + 1e-30)
                mu_then = oldest["mean_net_start"](full["observations"]).flatten()
                across["mu_efficiency_10"] = th.sqrt(((mu_end - mu_then) ** 2).mean()).item() / (
                    sum(h["mu_shift_abs"] for h in self._history) + 1e-30)
            self.diag_rounds.append(
                {
                    "round": round_index,
                    "clip_range": float(clip_range),
                    "n_epochs": self.n_epochs,
                    "sigma_start": last["sigma_start"],
                    "sigma_end": float(th.exp(self.policy.log_std).mean().item()),
                    "kl_exact_end": last["kl_exact"],
                    "kl_k3_full_end": last["kl_k3_full"],
                    "sb3_approx_kl": last["sb3_approx_kl_epoch"],
                    "mean_shift_rms_end": last["mean_shift_rms"],
                    "mu_shift_abs": mu_shift_abs,
                    "actor_path_length": path_length,
                    "actor_net_displacement": net,
                    "path_efficiency": net / (path_length + 1e-30),
                    **across,
                }
            )

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def save_diagnostics(self, run_dir) -> None:
        import pandas as pd

        pd.DataFrame(self.diag_epochs).to_csv(f"{run_dir}/diag_epochs.csv", index=False)
        pd.DataFrame(self.diag_rounds).to_csv(f"{run_dir}/diag_rounds.csv", index=False)
