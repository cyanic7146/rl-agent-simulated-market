import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# numbers pasted from the 1000 seed runs of run_experiment, crowding, information and interpret_rl
os.makedirs("PAPER_FIGURES", exist_ok=True)

agent_names = [
    "Random1", "Random2", "Random3",
    "Momentum1", "Momentum2", "Momentum3",
    "MeanReversion1", "MeanReversion2", "MeanReversion3",
    "BuyAndHold1", "BuyAndHold2", "BuyAndHold3",
    "Contrarian1", "Contrarian2", "Contrarian3",
    "VolumePressure1", "VolumePressure2", "VolumePressure3",
    "ValueEstimate1", "ValueEstimate2", "ValueEstimate3",
    "RiskManaged1", "RiskManaged2", "RiskManaged3",
    "RL agent",
]

# risky preset, same order as agent_names
risky_returns = [
    -1.36, -2.67, -4.24,
    1.38, 2.33, 2.41,
    -3.19, -3.34, -1.48,
    -0.44, -0.66, -0.92,
    -4.41, -4.83, -3.24,
    -4.17, -5.02, -5.44,
    -1.26, 0.43, 1.97,
    -1.68, -1.71, -1.43,
    9.98,
]

# rl bar here is the risky model visiting the calm market
calm_returns = [
    -0.71, -1.31, -1.90,
    0.73, 0.92, 0.43,
    0.75, 1.22, 1.54,
    -0.15, -0.23, -0.32,
    0.08, 0.28, 0.22,
    -0.23, -0.43, -0.56,
    1.73, 1.91, 1.46,
    0.82, 0.91, 1.17,
    1.72,
]


# plot zero, risky preset on its own
fig, ax = plt.subplots(figsize=(14, 6))
colors = []
for name in agent_names:
    colors.append("tab:red" if name == "RL agent" else "tab:gray")
ax.bar(np.arange(len(agent_names)), risky_returns, 0.7, color=colors, alpha=0.9)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(np.arange(len(agent_names)))
ax.set_xticklabels(agent_names, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("average return over 1000 seeds (%)")
ax.set_title("average return by agent, risky market")
fig.tight_layout()
fig.savefig("PAPER_FIGURES/returns_comparison.png", dpi=120)
plt.close(fig)


# plot a, both presets, rl only trained on risky
x = np.arange(len(agent_names))
width = 0.38

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width / 2, risky_returns, width, label="risky preset", color="tab:red", alpha=0.85)
ax.bar(x + width / 2, calm_returns, width, label="calm preset", color="tab:blue", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(agent_names, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("average return over 1000 seeds (%)")
ax.set_title("average return by agent, risky vs calm market")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("PAPER_FIGURES/returns_risky_vs_calm.png", dpi=120)
plt.close(fig)


# plot b, crowding. 20 seeds looked dramatic, 100 calmed it down, 1000 flattened it
copies = [1, 2, 4, 8]
per_copy_return = [9.98, 10.00, 9.55, 9.42]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(copies, per_copy_return, marker="o", color="tab:red", linewidth=2)
for c, r in zip(copies, per_copy_return):
    ax.annotate(f"{r:.1f}%", (c, r), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set_xticks(copies)
ax.set_xlabel("number of rl agent copies in the market")
ax.set_ylabel("per-copy average return (%)")
ax.set_title("crowding experiment, 1000 seeds per setting")
ax.set_ylim(0, 14)
fig.tight_layout()
fig.savefig("PAPER_FIGURES/crowding.png", dpi=120)
plt.close(fig)


# plot c, information experiment
informed_counts = [0, 1, 3, 6]
background_returns = [-1.79, -1.26, -0.98, -0.83]
informed_returns = [None, 24.58, 22.90, 23.61]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(informed_counts, background_returns, marker="o", color="gray", linewidth=2, label="background agents")
ax.plot(informed_counts[1:], informed_returns[1:], marker="s", color="tab:green", linewidth=2, label="informed agents")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(informed_counts)
ax.set_xlabel("number of informed agents")
ax.set_ylabel("average return over 1000 seeds (%)")
ax.set_title("information experiment, noisy peek at the fundamental")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("PAPER_FIGURES/information.png", dpi=120)
plt.close(fig)


# plot d, train/test matrix, both models in both markets
test_labels = ["tested in risky", "tested in calm"]
risky_trained = [9.98, 1.72]
calm_trained = [8.69, 4.04]

x = np.arange(len(test_labels))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5))
bars_risky = ax.bar(x - width / 2, risky_trained, width, label="risky-trained model", color="tab:red", alpha=0.85)
bars_calm = ax.bar(x + width / 2, calm_trained, width, label="calm-trained model", color="tab:blue", alpha=0.85)
for bar in list(bars_risky) + list(bars_calm):
    ax.annotate(f"{bar.get_height():.2f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(test_labels)
ax.set_ylabel("average return over 1000 seeds (%)")
ax.set_title("train/test matrix, each model in each market")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("PAPER_FIGURES/train_test_matrix.png", dpi=120)
plt.close(fig)


# plot d2, risky policy action mix per regime, the 96 idle episodes dropped first
action_labels = ["hold", "buy small", "buy medium", "buy large", "sell small", "sell medium", "sell large"]
by_regime = {
    "mean_reverting": [8.3, 20.0, 16.5, 7.6, 38.6, 0.4, 8.5],
    "trending": [7.9, 23.7, 19.0, 8.9, 31.4, 0.4, 8.7],
    "volatile": [8.0, 17.3, 28.9, 8.4, 25.9, 1.4, 10.2],
}

x = np.arange(len(action_labels))
width = 0.27

fig, ax = plt.subplots(figsize=(10, 5))
for offset, (regime, frequencies) in zip([-width, 0, width], by_regime.items()):
    ax.bar(x + offset, frequencies, width, label=regime, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(action_labels, rotation=20, ha="right")
ax.set_ylabel("share of steps in that regime (%)")
ax.set_title("risky policy action mix inside each hidden regime, 1000 episodes")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("PAPER_FIGURES/action_by_regime.png", dpi=120)
plt.close(fig)


# plot e, calm training curve from the eval history
if os.path.exists("MODELS/evaluations_calm.npz"):
    data = np.load("MODELS/evaluations_calm.npz")
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    best_index = mean_rewards.argmax()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timesteps / 1e6, mean_rewards, color="tab:blue", linewidth=1)
    ax.axvline(timesteps[best_index] / 1e6, color="tab:red", linestyle="--", linewidth=1.2,
               label=f"best checkpoint, {mean_rewards[best_index]:.3f} at {timesteps[best_index] / 1e6:.2f}M steps")
    ax.set_xlabel("training timesteps (millions)")
    ax.set_ylabel("mean eval reward (50 episodes)")
    ax.set_title("calm model training curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("PAPER_FIGURES/training_curve_calm.png", dpi=120)
    plt.close(fig)


print("paper figures saved to PAPER_FIGURES/")
print("  PAPER_FIGURES/returns_comparison.png")
print("  PAPER_FIGURES/returns_risky_vs_calm.png")
print("  PAPER_FIGURES/crowding.png")
print("  PAPER_FIGURES/information.png")
print("  PAPER_FIGURES/train_test_matrix.png")
print("  PAPER_FIGURES/action_by_regime.png")
