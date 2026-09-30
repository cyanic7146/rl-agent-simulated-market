import os
import numpy as np

from values import VALUES
from market_env import MarketEnv


# action number -> name
def describe_action(action):
    names = {0: "hold", 1: "buy_small", 2: "buy_medium", 3: "buy_large", 4: "sell_small", 5: "sell_medium", 6: "sell_large"}
    return names[action]


model_path = "MODELS/best_model_31obs_checkpoint/best_model"
# model_path = "MODELS/best_model_calm_31obs_checkpoint/best_model"
normalizer_path = "MODELS/vec_normalize.pkl"
# normalizer_path = "MODELS/vec_normalize_calm.pkl"

if not (os.path.exists(model_path + ".zip") and os.path.exists(normalizer_path)):
    print("no trained model found in MODELS/, nothing to interpret")
else:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    model = PPO.load(model_path)
    norm_loader = DummyVecEnv([MarketEnv])
    norm_loader = VecNormalize.load(normalizer_path, norm_loader)
    obs_mean = norm_loader.obs_rms.mean
    obs_var = norm_loader.obs_rms.var
    obs_clip = norm_loader.clip_obs

    env = MarketEnv()
    num_episodes = 1000 # lower this for a quick look
    seeds = list(range(1, num_episodes + 1))

    records = [] # one row per step, every episode

    idle = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)
        price_trail = []
        episode = []
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            position_before = env.rl_agent.position

            obs, reward, terminated, truncated, info = env.step(action)
            obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)

            price = env.market.price
            entry = env.market.history[-1]
            position_after = env.rl_agent.position

            episode.append({
                "seed": seed,
                "step": entry["step"],
                "price": price,
                "action": action,
                "position": position_after,
                "portfolio": info["portfolio_value"],
                "regime": entry["regime"],
                "trade": position_after - position_before,
                "prev_prices": price_trail[-5:],
            })

            price_trail.append(price)

        # skip idle episodes, the portfolio never moved so its the same failure counted every step
        portfolios = [r["portfolio"] for r in episode]
        if max(portfolios) == min(portfolios):
            idle += 1
            continue
        records.extend(episode)

    total = len(records)
    print(f"\nlogged {total} steps over {len(seeds) - idle} traded episodes ({idle} idle ones dropped)")

    # a, overall action frequency
    print("\noverall action frequency:")
    overall_counts = {}
    for r in records:
        overall_counts[r["action"]] = overall_counts.get(r["action"], 0) + 1
    for action in range(7):
        count = overall_counts.get(action, 0)
        print(f"  {describe_action(action)}: {count} ({100 * count / total:.1f}%)")

    # b, split by regime
    regimes = ["mean_reverting", "trending", "volatile"]
    print("\naction frequency by regime:")
    for regime in regimes:
        regime_records = [r for r in records if r["regime"] == regime]
        regime_total = len(regime_records)
        print(f"  {regime} ({regime_total} steps):")
        if regime_total == 0:
            continue
        counts = {}
        for r in regime_records:
            counts[r["action"]] = counts.get(r["action"], 0) + 1
        for action in range(7):
            count = counts.get(action, 0)
            print(f"    {describe_action(action)}: {count} ({100 * count / regime_total:.1f}%)")

    # c, biggest trades and what the price was doing right before
    print("\ntop 5 biggest buys (price over the 5 steps before):")
    biggest_buys = sorted([r for r in records if r["trade"] > 0], key=lambda r: r["trade"], reverse=True)[:5]
    for r in biggest_buys:
        before = ", ".join(f"{p:.2f}" for p in r["prev_prices"])
        print(f"  +{r['trade']} shares at {r['price']:.2f} in {r['regime']} before: [{before}]")

    print("\ntop 5 biggest sells (price over the 5 steps before):")
    biggest_sells = sorted([r for r in records if r["trade"] < 0], key=lambda r: r["trade"])[:5]
    for r in biggest_sells:
        before = ", ".join(f"{p:.2f}" for p in r["prev_prices"])
        print(f"  {r['trade']} shares at {r['price']:.2f} in {r['regime']} before: [{before}]")

    # d, avg position in each regime
    print("\naverage position held by regime:")
    for regime in regimes:
        regime_records = [r for r in records if r["regime"] == regime]
        if len(regime_records) == 0:
            print(f"  {regime}: no steps")
            continue
        average_position = sum(r["position"] for r in regime_records) / len(regime_records)
        print(f"  {regime}: {average_position:.1f}")
