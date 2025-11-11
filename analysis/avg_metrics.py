import json

import numpy as np

# List of seeds
seeds = [0, 1, 12345]
base_paths = [
    "./experiment-results/femnist-cnn",
    "./experiment-results/cifar10-cnn-cifar",
    "./experiment-results/shakespeare-lstm",
]

# Store all best metrics
all_best_metrics = []
best_rounds = {}

for seed in seeds:
    file_path = f"{base_paths[0]}/fedavg-results-seed-{seed}.json"

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

print("\n--- Average metrics across seeds ---")

# Collect all metric names
metric_names = all_best_metrics[0].keys()

# Compute average per metric
avg_metrics = {}
for metric in metric_names:
    values = [
        m[metric] for m in all_best_metrics if isinstance(m[metric], (int, float))
    ]
    avg_metrics[metric] = float(np.mean(values)) if values else None

# Print average metrics
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.6f}" if value is not None else f"{metric}: N/A")

print("\n--- Best rounds per seed ---")
for seed, round_ in best_rounds.items():
    print(f"Seed {seed}: Round {round_}")
