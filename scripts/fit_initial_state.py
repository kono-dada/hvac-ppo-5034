import numpy as np
from hvac.env import RoomConfig, RoomThermalEnv
from hvac.controllers import ThermostatController, solve_oracle
from hvac.evaluation import eval_seeds
env = RoomThermalEnv()
rows = []
for ctrl_name in ("thermostat2h", "oracle"):
    pts = []
    for s in eval_seeds(60):
        x = None
        for day in range(6):
            _, info = env.reset(seed=s)
            if x is not None:
                env.x = x.copy()
            if ctrl_name == "oracle":
                plan = solve_oracle(env)
            c = ThermostatController(prestart_hours=2)
            for k in range(96):
                a = plan[k] if ctrl_name == "oracle" else c.act(env)
                env.step(np.array([a], dtype=np.float32))
            x = env.x.copy()
        pts.append((info["t_out_mean"], x[1], x[0]))
    pts = np.array(pts)
    A = np.c_[pts[:, 0] - 22.5, np.ones(len(pts))]
    coef, *_ = np.linalg.lstsq(A, pts[:, 1] - 22.5, rcond=None)
    resid = pts[:, 1] - 22.5 - A @ coef
    print(ctrl_name, "T_mass(midnight) - 22.5 = %.3f * (T_mean - 22.5) + %.2f, resid std %.2f" % (coef[0], coef[1], resid.std()))
    print("   air - mass at midnight: mean %.2f std %.2f" % ((pts[:, 2] - pts[:, 1]).mean(), (pts[:, 2] - pts[:, 1]).std()))
    for lo, hi in ((-2, 5), (5, 12), (12, 19), (19, 26), (26, 32)):
        m = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        print(f"   T_mean {lo:3d}..{hi:3d}: T_mass midnight {pts[m,1].mean():.2f} (n={m.sum()})")
