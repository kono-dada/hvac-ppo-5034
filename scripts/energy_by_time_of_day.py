import glob, numpy as np
from stable_baselines3 import PPO
from hvac.env import RoomConfig, RoomThermalEnv
from hvac.evaluation import eval_seeds, run_episode
cfg = RoomConfig(); env = RoomThermalEnv(cfg); seeds = eval_seeds(100)
occ = env.occupied[:96]; hrs = env.hours[:96]
def split(policy=None, oracle=False):
    pre, day, eve = [], [], []
    for s in seeds:
        r = run_episode(env, s, policy=policy, oracle=oracle, record=True)
        e = r["trajectory"]["energy_kwh"]
        pre.append(e[hrs < 8].sum()); day.append(e[occ].sum()); eve.append(e[hrs >= 18].sum())
    return np.mean(pre), np.mean(day), np.mean(eve)
print("energy kWh/day split:   00-08   08-18   18-24")
print("oracle           %6.2f %6.2f %6.2f" % split(oracle=True))
for v in ("baseline", "end_truncate"):
    rows = []
    for d in sorted(glob.glob(f"runs/pre/{v}/seed*")):
        m = PPO.load(d + "/model", device="cpu")
        rows.append(split(policy=lambda _e, o: float(m.predict(o, deterministic=True)[0][0])))
    rows = np.array(rows)
    print(f"{v:16s} %6.2f %6.2f %6.2f" % tuple(rows.mean(0)), " per-seed 18-24:", np.round(rows[:, 2], 2))
