import os
import numpy as np

from values import VALUES
from market_env import MarketEnv


# run_experiment.py only prints averages, is the rl average just a few lucky seeds?
if not (os.path.exists("MODELS/best_model_31obs_checkpoint/best_model.zip") and os.path.exists("MODELS/vec_normalize.pkl")):
    print("no trained model found in MODELS/, nothing to run")
else:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    model = PPO.load("MODELS/best_model_31obs_checkpoint/best_model")
    norm_loader = DummyVecEnv([MarketEnv])
    norm_loader = VecNormalize.load("MODELS/vec_normalize.pkl", norm_loader)
    obs_mean = norm_loader.obs_rms.mean
    obs_var = norm_loader.obs_rms.var
    obs_clip = norm_loader.clip_obs

    env = MarketEnv()
    seeds = list(range(1, 1001))
    starting_cash = VALUES["initial_cash"]

    finals = []
    bah_finals = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)
        finals.append(info["portfolio_value"])
        bah_finals.append(env.bah_agent.portfolio_value(env.market.price))

    returns = (np.array(finals) - starting_cash) / starting_cash * 100
    diffs = np.array(finals) - np.array(bah_finals)

    print(f"per-seed rl returns over {len(seeds)} seeds")
    # only dump every seed for small runs
    if len(seeds) <= 50:
        for start in range(0, len(seeds), 10):
            chunk = " ".join(f"{r:.2f}" for r in returns[start:start + 10])
            print(f"  seeds {seeds[start]}-{seeds[start] + 9}: {chunk}")

    print(f"\nmean={returns.mean():.2f}% std={returns.std():.2f}%")
    standard_error = returns.std(ddof=1) / np.sqrt(len(seeds))
    print(f"standard error of the mean={standard_error:.3f} 95% ci=[{returns.mean() - 1.96 * standard_error:.2f}%, {returns.mean() + 1.96 * standard_error:.2f}%]")
    print(f"seeds sitting in cash the whole episode: {int((returns == 0).sum())}/{len(seeds)}")
    print(f"min={returns.min():.2f}% q25={np.percentile(returns, 25):.2f}% median={np.percentile(returns, 50):.2f}% q75={np.percentile(returns, 75):.2f}% max={returns.max():.2f}%")
    print(f"seeds with negative return: {int((returns < 0).sum())}/{len(seeds)}")
    print(f"seeds behind in-episode buy and hold: {int((diffs <= 0).sum())}/{len(seeds)}")

    #worst 5
    worst_order = np.argsort(returns)[:5]
    print("\nworst 5 seeds:")
    for i in worst_order:
        print(f"  seed {seeds[i]}: return={returns[i]:.2f}% vs in-episode bah diff={diffs[i]:.2f}")
