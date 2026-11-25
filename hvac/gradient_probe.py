"""Same-minibatch gradient and same-Adam-state probes along a clip=0.5 run."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch as th

from hvac.diagnostics import DiagnosticPPO, _clip_gradient_parts, _consistency, _flat


def _cos(a: th.Tensor, b: th.Tensor) -> float:
    return float((a @ b / (a.norm() * b.norm() + 1e-30)).item())


class ClipGradientProbePPO(DiagnosticPPO):
    def __init__(self, *args, compare_range=0.2, probe_rounds=(71, 100), probe_every=16, **kwargs):
        self.compare_range = compare_range
        self.probe_rounds = probe_rounds
        self.probe_every = probe_every
        self.gradient_epochs = []
        self.adam_probes = []
        self._probe_key = None
        self._probe_vectors = {k: [] for k in ("g2", "g5", "h2", "h5")}
        self._probe_metrics = []
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "gradient_epochs", "adam_probes", "_probe_key", "_probe_vectors", "_probe_metrics",
        ]

    @th.no_grad()
    def _flush_gradient_epoch(self):
        if not self._probe_metrics:
            return
        row = {"round": self._probe_key[0], "epoch": self._probe_key[1]}
        for col in self._probe_metrics[0]:
            row[col] = float(np.mean([m[col] for m in self._probe_metrics]))
        for label in ("g2", "g5", "h2", "h5"):
            vs = self._probe_vectors[label]
            row[f"consistency_{label}"], row[f"cos_to_mean_{label}"] = _consistency(vs)
            row[f"mean_norm_{label}"] = th.stack(vs).mean(0).norm().item()
        for a, b, label in (("g2", "g5", "raw"), ("h2", "h5", "norm_clipped")):
            m2 = th.stack(self._probe_vectors[a]).mean(0)
            m5 = th.stack(self._probe_vectors[b]).mean(0)
            added = m5 - m2
            row[f"added_mean_proj_{label}"] = (added @ m5 / (m5.square().sum() + 1e-30)).item()
            row[f"added_mean_norm_share_{label}"] = (added.norm() / (m5.norm() + 1e-30)).item()
            row[f"mean_gradient_cos_{label}"] = _cos(m2, m5)
            row[f"mean_norm_ratio_{label}"] = (m5.norm() / (m2.norm() + 1e-30)).item()
            # Exclude log_std to isolate gradients of the action-mean network.
            row[f"added_mu_mean_proj_{label}"] = (added[1:] @ m5[1:] / (m5[1:].square().sum() + 1e-30)).item()
        self.gradient_epochs.append(row)
        self._probe_vectors = {k: [] for k in self._probe_vectors}
        self._probe_metrics = []

    def _prepare_gradient_probe(self, **ctx):
        key = (ctx["round"], ctx["epoch"])
        if key != self._probe_key:
            self._flush_gradient_epoch()
            self._probe_key = key
        a, r = ctx["advantages"], ctx["ratio"]
        loss2 = -th.min(a * r, a * th.clamp(r, 1 - self.compare_range, 1 + self.compare_range)).mean()
        g2 = _flat(th.autograd.grad(loss2, ctx["actor_params"], retain_graph=True))
        with th.no_grad():
            w2 = _clip_gradient_parts(a, r, self.compare_range)["weight"]
            w5 = _clip_gradient_parts(a, r, ctx["clip_range"])["weight"]
            extra_frac = (w2 > w5).float().mean().item()
            mu_reference = ctx["mean_net_start"](ctx["observations"]).flatten().clone()
        mu_current = self._mean_net()(ctx["observations"]).flatten()
        distance = 0.5 * (mu_current - mu_reference).square().mean()
        distance_grads = th.autograd.grad(distance, ctx["actor_params"], allow_unused=True)
        distance_grad = _flat([g if g is not None else th.zeros_like(p)
                               for g, p in zip(distance_grads, ctx["actor_params"])])
        return {**ctx, "g2": g2, "extra_frac": extra_frac,
                "distance_grad": distance_grad, "mu_reference": mu_reference}

    @th.no_grad()
    def _before_gradient_probe_step(self, probe, actor_params, critic_params, g_c):
        g2 = probe["g2"]
        actor_grads = {}
        off = 0
        for p in actor_params:
            actor_grads[id(p)] = g2[off:off + p.numel()].reshape_as(p)
            off += p.numel()
        all_params = list(self.policy.parameters())
        grads2 = [actor_grads.get(id(p), p.grad) for p in all_params]
        total2 = th.linalg.vector_norm(th.stack([th.linalg.vector_norm(g) for g in grads2 if g is not None]))
        scale2 = (self.max_grad_norm / (total2 + 1e-6)).clamp(max=1)
        probe["h2"] = g2 * scale2
        probe["g5"] = g_c
        probe["theta_before"] = _flat(actor_params).clone()
        probe["metrics"] = {
            "extra_samples_frac": probe["extra_frac"],
            "norm_g2": g2.norm().item(), "norm_g5": g_c.norm().item(),
            "added_norm_share": ((g_c - g2).norm() / (g_c.norm() + 1e-30)).item(),
            "added_proj_g5": ((g_c - g2) @ g_c / (g_c.square().sum() + 1e-30)).item(),
            "gradient_cos_g2_g5": _cos(g2, g_c),
            "scale2": scale2.item(),
            "restore_dot_g02": (g2 @ probe["distance_grad"]).item(),
            "restore_dot_g05": (g_c @ probe["distance_grad"]).item(),
            "restore_dot_h02": (probe["h2"] @ probe["distance_grad"]).item(),
            "restore_cos_g02": _cos(g2, probe["distance_grad"]),
            "restore_cos_g05": _cos(g_c, probe["distance_grad"]),
        }
        directions = []
        opt = self.policy.optimizer
        beta2, eps = opt.defaults["betas"][1], opt.defaults["eps"]
        for p in actor_params:
            state = opt.state.get(p, {})
            if not state:
                directions.append(th.zeros_like(p))
            else:
                n = state["step"].item()
                denom = state["exp_avg_sq"].sqrt() + eps * (1 - beta2 ** n) ** 0.5
                directions.append(-state["exp_avg"] / (denom + 1e-30))
        probe["metrics"]["momentum_outward_cos"] = _cos(_flat(directions), probe["distance_grad"])
        if not (self.probe_rounds[0] <= probe["round"] <= self.probe_rounds[1]
                and probe["minibatch"] % self.probe_every == 0):
            return
        # An independent optimizer with cloned parameters, identical state, and
        # the counterfactual actor gradient. Critic gradients are held fixed.
        cloned = {id(p): th.nn.Parameter(p.detach().clone()) for p in all_params}
        opt = self.policy.optimizer
        groups = [{**{k: v for k, v in group.items() if k != "params"},
                   "params": [cloned[id(p)] for p in group["params"]]} for group in opt.param_groups]
        shadow_opt = type(opt)(groups, **opt.defaults)
        shadow_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        for p, g in zip(all_params, grads2):
            cloned[id(p)].grad = g.detach().clone() if g is not None else None
        th.nn.utils.clip_grad_norm_(list(cloned.values()), self.max_grad_norm)
        shadow_h2 = _flat([cloned[id(p)].grad for p in actor_params])
        assert th.allclose(shadow_h2, probe["h2"], rtol=1e-5, atol=1e-7)
        shadow_opt.step()
        probe["step2"] = _flat([cloned[id(p)] for p in actor_params]) - probe["theta_before"]
        probe["mu_before"] = self._mean_net()(probe["observations"]).flatten().clone()
        shadow_mean = copy.deepcopy(self._mean_net())
        for q, p in zip(shadow_mean.parameters(), actor_params[1:]):
            assert q.shape == p.shape
            q.copy_(cloned[id(p)])
        probe["mu_step2"] = shadow_mean(probe["observations"]).flatten() - probe["mu_before"]

    @th.no_grad()
    def _after_gradient_probe_step(self, probe, actor_params):
        g2, g5, h2 = probe["g2"], probe["g5"], probe["h2"]
        h5 = _flat([p.grad for p in actor_params]).clone()
        step5 = _flat(actor_params) - probe["theta_before"]
        metrics = probe["metrics"]
        metrics["added_cos_to_actual_descent"] = _cos(-(g5 - g2), step5)
        metrics["restore_dot_h05"] = (h5 @ probe["distance_grad"]).item()
        metrics["step_outward_cos"] = _cos(step5, probe["distance_grad"])
        self._probe_metrics.append(metrics)
        for name, v in (("g2", g2), ("g5", g5), ("h2", h2), ("h5", h5)):
            self._probe_vectors[name].append(v.detach().clone())
        if "step2" not in probe:
            return
        step2 = probe["step2"]
        mu5 = self._mean_net()(probe["observations"]).flatten() - probe["mu_before"]
        mu2 = probe["mu_step2"]
        self.adam_probes.append({
            "round": probe["round"], "epoch": probe["epoch"], "minibatch": probe["minibatch"],
            "extra_samples_frac": probe["extra_frac"],
            "param_step_02": step2.norm().item(), "param_step_05": step5.norm().item(),
            "param_step_cos": _cos(step2, step5),
            "param_extra_proj": ((step5 - step2) @ step5 / (step5.square().sum() + 1e-30)).item(),
            "mu_step_rms_02": mu2.square().mean().sqrt().item(),
            "mu_step_rms_05": mu5.square().mean().sqrt().item(),
            "mu_step_cos": _cos(mu2, mu5),
            "mu_extra_proj": ((mu5 - mu2) @ mu5 / (mu5.square().sum() + 1e-30)).item(),
            "added_cos_to_actual_descent": metrics["added_cos_to_actual_descent"],
            "distance_delta_02": (0.5 * ((probe["mu_before"] + mu2 - probe["mu_reference"]).square()
                                         - (probe["mu_before"] - probe["mu_reference"]).square()).mean()).item(),
            "distance_delta_05": (0.5 * ((probe["mu_before"] + mu5 - probe["mu_reference"]).square()
                                         - (probe["mu_before"] - probe["mu_reference"]).square()).mean()).item(),
        })

    def save_diagnostics(self, run_dir):
        import pandas as pd
        self._flush_gradient_epoch()
        super().save_diagnostics(run_dir)
        run_dir = Path(run_dir)
        pd.DataFrame(self.gradient_epochs).to_csv(run_dir / "gradient_epochs.csv", index=False)
        pd.DataFrame(self.adam_probes).to_csv(run_dir / "adam_probes.csv", index=False)
