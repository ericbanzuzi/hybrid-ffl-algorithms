import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Config ---
# 0: FEMNIST, 1: CIFAR-10, 2: Shakespeare, 3: Fashion-MNIST, 4: CIFAR-10-CD
experiment = 0
seeds = [0, 1, 12345]
base_paths = [
    "./experiment-results/femnist-cnn",
    "./experiment-results/cifar10-cnn-cifar",
    "./experiment-results/shakespeare-lstm-vm",
    "./experiment-results/fashion-mnist-cnn",
    "./experiment-results/cifar10-cnn-cifar-cd",
]

sns.set(style="whitegrid", font_scale=1.2)

algorithms = {
    "FedAvg": "fedavg-results-seed-{}.json",
    "q-FedAvg": "qfedavg-results-seed-{}.json",
    "FedProx": "fedprox-results-seed-{}.json",
    "FedYogi": "fedyogi-fedavg-results-seed-{}.json",
    "Ditto": "fedavg-ditto-results-seed-{}.json",
    "FedProxYogi": "fedyogi-fedprox-results-seed-{}.json",
    "FedProxDitto": "fedavg-fedproxditto-results-seed-{}.json",
    "FedYogiDitto": "fedyogi-ditto-results-seed-{}.json",
    "FedProxYogiDitto": "fedyogi-fedproxditto-results-seed-{}.json",
}


plt.figure(figsize=(10, 6))

for algo, pattern in algorithms.items():
    all_accuracies = {}

    # --- Load results for all seeds ---
    for seed in seeds:
        file_path = os.path.join(base_paths[experiment], pattern.format(seed))
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
    plt.plot(rounds, mean_acc, label=algo, linewidth=1.5)
    plt.fill_between(
        rounds,
        np.array(mean_acc) - np.array(std_acc),
        np.array(mean_acc) + np.array(std_acc),
        alpha=0.2,
    )

# --- Labels and styling ---
titles = [
    "FEMNIST",
    "CIFAR-10 - cross-silo",
    "Shakespeare",
    "Fashion-MNIST",
    "CIFAR-10 - cross-device",
]
plt.title(f"Worst 10% Accuracy per Round ({titles[experiment]})")
plt.xlabel("Round")
plt.ylabel("Average Test Accuracy")
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
plt.tight_layout()
plt.show()
