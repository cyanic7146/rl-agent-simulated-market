import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from values import VALUES
from market_env import MarketEnv


# turns the action number into something readable
def describe_action(action):
    if action == 1:
        return "buy", "small"
    if action == 2:
        return "buy", "medium"
    if action == 3:
        return "buy", "large"
    if action == 4:
        return "sell", "small"
    if action == 5:
        return "sell", "medium"
    if action == 6:
        return "sell", "large"
    return "hold", "none"


os.makedirs("PLOTS", exist_ok=True)

# one run gives all three plots, same 24 background agents as main.py
env = MarketEnv()

model = None
obs_mean = None
obs_var = None
obs_clip = 10.0
if os.path.exists("MODELS/best_model_31obs_checkpoint/best_model.zip") and os.path.exists("MODELS/vec_normalize.pkl"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    model = PPO.load("MODELS/best_model_31obs_checkpoint/best_model")
    norm_loader = DummyVecEnv([MarketEnv])
    norm_loader = VecNormalize.load("MODELS/vec_normalize.pkl", norm_loader)
    obs_mean = norm_loader.obs_rms.mean
    obs_var = norm_loader.obs_rms.var
    obs_clip = norm_loader.clip_obs


obs, _ = env.reset(seed=VALUES["seed"])

agent_names = [agent.name for agent in env.market.agents]
portfolio_curves = {name: [] for name in agent_names}
prices = []
fundamentals = []
regimes = []
rl_actions = []

step_index = 0
terminated = False
while not terminated:
    if model is not None:
        norm_obs = np.clip((obs - obs_mean) / np.sqrt(obs_var + 1e-8), -obs_clip, obs_clip).astype(np.float32)
        action, _ = model.predict(norm_obs, deterministic=True)
        action = int(action)
    else:
        action = 0

    obs, reward, terminated, truncated, info = env.step(action)

    price = env.market.price
    last_entry = env.market.history[-1]
    prices.append(price)
    fundamentals.append(last_entry["fundamental_value"])
    regimes.append(last_entry["regime"])

    for agent in env.market.agents:
        portfolio_curves[agent.name].append(agent.portfolio_value(price))

    direction, size = describe_action(action)
    if direction != "hold":
        rl_actions.append((step_index, price, direction, size))

    step_index += 1


steps = list(range(len(prices)))

# group steps with the same regime so each block can be shaded
regime_colors = {"mean_reverting": "tab:blue", "trending": "tab:orange", "volatile": "tab:red"}
regime_spans = []
span_start = 0
for i in range(1, len(regimes)):
    if regimes[i] != regimes[i-1]:
        regime_spans.append((span_start, i, regimes[i-1]))
        span_start = i
regime_spans.append((span_start, len(regimes), regimes[-1]))


# plot a, price vs fundamental with regime shading
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(steps, prices, label="market price", color="black", linewidth=1.5)
ax.plot(steps, fundamentals, label="fundamental value", color="green", linewidth=1.2, linestyle="--")
already_labeled = set()
for start, end, regime in regime_spans:
    if regime not in already_labeled:
        label = regime
    else:
        label = None
    already_labeled.add(regime)
    ax.axvspan(start, end, color=regime_colors.get(regime, "gray"), alpha=0.12, label=label)
ax.set_xlabel("step")
ax.set_ylabel("price")
ax.set_title("market price vs fundamental value, shaded by regime")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("PLOTS/price_vs_fundamental.png", dpi=120)
plt.close(fig)


# plot b, every agents portfolio, rl in red
fig, ax = plt.subplots(figsize=(12, 6))
for name in agent_names:
    if name == "RLAgent":
        continue
    ax.plot(steps, portfolio_curves[name], color="gray", alpha=0.4, linewidth=0.8)
if model is not None:
    ax.plot(steps, portfolio_curves["RLAgent"], color="red", linewidth=2.2, label="RLAgent")
    ax.legend(loc="best")
ax.set_xlabel("step")
ax.set_ylabel("portfolio value")
ax.set_title("agent portfolio value over time")
fig.tight_layout()
fig.savefig("PLOTS/portfolios.png", dpi=120)
plt.close(fig)


# plot c, rl trades on the price chart
buy_colors = {"small": "#9be29b", "medium": "#3fae3f", "large": "#0a6b0a"}
sell_colors = {"small": "#f0a0a0", "medium": "#d83f3f", "large": "#8b0000"}

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(steps, prices, color="black", linewidth=1.2)
for step_i, price, direction, size in rl_actions:
    if direction == "buy":
        ax.scatter(step_i, price, marker="^", color=buy_colors[size], s=70, zorder=3)
    else:
        ax.scatter(step_i, price, marker="v", color=sell_colors[size], s=70, zorder=3)

legend_handles = [
    Line2D([0], [0], color="black", linewidth=1.2, label="market price"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=buy_colors["small"], markersize=10, label="buy small"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=buy_colors["medium"], markersize=10, label="buy medium"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=buy_colors["large"], markersize=10, label="buy large"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor=sell_colors["small"], markersize=10, label="sell small"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor=sell_colors["medium"], markersize=10, label="sell medium"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor=sell_colors["large"], markersize=10, label="sell large"),
]
ax.legend(handles=legend_handles, loc="best")
ax.set_xlabel("step")
ax.set_ylabel("price")
ax.set_title("rl agent trades on the price chart")
fig.tight_layout()
fig.savefig("PLOTS/rl_actions.png", dpi=120)
plt.close(fig)


print("plots saved to PLOTS/")
print("  PLOTS/price_vs_fundamental.png")
print("  PLOTS/portfolios.png")
print("  PLOTS/rl_actions.png")
if model is None:
    print("no trained model found, rl trade plot will be empty and rl portfolio not highlighted")
