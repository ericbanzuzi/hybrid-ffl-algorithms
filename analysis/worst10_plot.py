import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Config ---
seeds = [0, 1, 12345]
base_path = "./experiment-results/femnist-cnn"

sns.set(style="whitegrid", font_scale=1.2)

algorithms = {
    "FedAvg": "fedavg-results-seed-{}.json",
    "q-FedAvg": "qfedavg-results-seed-{}.json",
}

plt.figure(figsize=(10, 6))

for algo, pattern in algorithms.items():
    all_accuracies = {}

    # --- Load results for all seeds ---
    for seed in seeds:
        file_path = os.path.join(base_path, pattern.format(seed))
        with open(file_path, "r") as f:
            results = json.load(f)

        for round_str, metrics in results.items():
            rnd = int(round_str)
            acc = metrics.get("worst-10", np.nan)
            all_accuracies.setdefault(rnd, []).append(acc)

    # --- Compute mean and std across seeds ---
    rounds = sorted(all_accuracies.keys())
    mean_acc = [np.nanmean(all_accuracies[r]) for r in rounds]
    std_acc = [np.nanstd(all_accuracies[r]) for r in rounds]

    # --- Plot mean + shaded std ---
    plt.plot(rounds, mean_acc, label=algo, linewidth=2)
    plt.fill_between(
        rounds,
        np.array(mean_acc) - np.array(std_acc),
        np.array(mean_acc) + np.array(std_acc),
        alpha=0.2,
    )

# --- Labels and styling ---
plt.title("Average Test Accuracy per Round (FEMNIST)")
plt.xlabel("Round")
plt.ylabel("Average Test Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
