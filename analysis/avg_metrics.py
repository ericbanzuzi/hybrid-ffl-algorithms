import json

import numpy as np

# List of seeds
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

# Store all best metrics
all_best_metrics = []
best_rounds = {}

for seed in seeds:
    file_path = f"{base_paths[experiment]}/fedavg-fedproxditto-results-seed-{seed}.json"

    # Load JSON file
    with open(file_path, "r") as f:
        results = json.load(f)

    # Find round (key) with highest avg-test-accuracy
    best_key = max(
        results, key=lambda k: results[k].get("avg-test-accuracy", float("-inf"))
    )
    best_metrics = results[best_key]

    # Record
    best_rounds[seed] = int(best_key)
    all_best_metrics.append(best_metrics)

    # Print per-seed summary
    print(
        f"Seed {seed} → Best round: {best_key}, avg-test-accuracy: {best_metrics['avg-test-accuracy']:.4f}"
    )

print("\n--- Average metrics across seeds (with standard errors) ---")

# Collect all metric names
metric_names = all_best_metrics[0].keys()

avg_metrics = {}
std_metrics = {}
se_metrics = {}

for metric in metric_names:
    # Extract numerical values only
    values = [
        m[metric] for m in all_best_metrics if isinstance(m[metric], (int, float))
    ]

    if not values:
        avg_metrics[metric] = None
        std_metrics[metric] = None
        se_metrics[metric] = None
        continue

    values = np.array(values)
    avg = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0  # sample std
    se = std / np.sqrt(len(values)) if len(values) > 1 else 0.0

    avg_metrics[metric] = avg
    std_metrics[metric] = std
    se_metrics[metric] = se

    print(f"{metric}:")
    print(f"  mean: {avg:.6f}")
    print(f"  std:  {std:.6f}")
    print(f"  se:   {se:.6f}")

print("\n--- Best rounds per seed ---")
for seed, round_ in best_rounds.items():
    print(f"Seed {seed}: Round {round_}")
