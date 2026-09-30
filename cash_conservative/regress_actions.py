import os
import numpy as np

from market_env import MarketEnv


# which signals actually drive the policy, not just eyeballing the biggest trades
feature_names = [
    "price_ratio", "position_fill", "cash_fill", "portfolio_level",
    "order_imbalance", "short_momentum", "long_momentum",
    "short_ma_gap", "long_ma_gap", "recent_volatility",
]
# where those sit in the 31 long obs, the 20 history ratios stay out
# above_start stays out too, its just portfolio_level minus one
feature_indices = [0, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# signed size so buys are + and sells are -
action_intent = {0: 0.0, 1: 0.25, 2: 0.75, 3: 1.5, 4: -0.25, 5: -0.75, 6: -1.5}


# ols with standard errors the long way
def ols(X, y):
    design = np.column_stack([np.ones(len(y)), X])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    dof = len(y) - design.shape[1]
    sigma_squared = residuals @ residuals / dof
    covariance = sigma_squared * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    ss_res = residuals @ residuals
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot
    return beta, standard_errors, r_squared


model_path = "MODELS/best_model_31obs_checkpoint/best_model"
# model_path = "MODELS/best_model_calm_31obs_checkpoint/best_model"
normalizer_path = "MODELS/vec_normalize.pkl"
# normalizer_path = "MODELS/vec_normalize_calm.pkl"

if not (os.path.exists(model_path + ".zip") and os.path.exists(normalizer_path)):
    print("no trained model found in MODELS/, nothing to regress")
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
    num_episodes = 1000 # keep the same as interpret_rl.py
    seeds = list(range(1, num_episodes + 1))

    raw_observations = []
    intents = []

    idle = 0
    for seed in seeds:
        raw_obs, _ = env.reset(seed=seed)
        episode_obs = []
        episode_intents = []
        portfolios = []
        terminated = False
        while not terminated:
            norm_obs = np.clip((raw_obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)
            action, _ = model.predict(norm_obs, deterministic=True)
            action = int(action)
            episode_obs.append(np.array(raw_obs, dtype=np.float64))
            episode_intents.append(action_intent[action])
            raw_obs, reward, terminated, truncated, info = env.step(action)
            portfolios.append(info["portfolio_value"])
        # same idle episode skip as interpret_rl.py
        if max(portfolios) == min(portfolios):
            idle += 1
            continue
        raw_observations.extend(episode_obs)
        intents.extend(episode_intents)

    X_full = np.array(raw_observations)
    y = np.array(intents)
    print(f"logged {len(y)} steps over {len(seeds) - idle} traded episodes ({idle} idle ones dropped)")

    # standardize so the coefs are comparable
    X = X_full[:, feature_indices]
    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    beta, standard_errors, r_squared = ols(X_standardized, y)

    print(f"\nols of signed action size on {len(feature_names)} standardized summary features")
    print(f"r squared: {r_squared:.3f}")
    print("\nsorted by |t|:")
    rows = []
    for i, name in enumerate(feature_names):
        coefficient = beta[i + 1]
        se = standard_errors[i + 1]
        rows.append((name, coefficient, se, coefficient / se))
    rows.sort(key=lambda row: abs(row[3]), reverse=True)
    for name, coefficient, se, t_value in rows:
        print(f"  {name}: coef={coefficient:.4f} se={se:.4f} t={t_value:.1f}")

    # r squared with every feature in, just to see what the raw history adds
    design_full = np.column_stack([np.ones(len(y)), X_full])
    beta_full, _, _, _ = np.linalg.lstsq(design_full, y, rcond=None)
    residuals_full = y - design_full @ beta_full
    r_squared_full = 1 - (residuals_full @ residuals_full) / ((y - y.mean()) ** 2).sum()
    print(f"\nr squared with all 31 features: {r_squared_full:.3f}")
