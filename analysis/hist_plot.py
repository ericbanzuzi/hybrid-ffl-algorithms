import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Configuration ---
seeds = [0, 1, 12345]
base_path = "./accuracies"

# best rounds per algorithm
best_rounds = {
    "fedavg": {0: 459, 1: 488, 12345: 453},
    "qfedavg": {0: 496, 1: 500, 12345: 479},
}

# --- Plot style ---
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))

# --- Load and plot both algorithms ---
for algo_name, rounds_dict in best_rounds.items():
    all_accuracies = []

    for seed in seeds:
        file_path = f"{base_path}/{algo_name}-femnist-accs-seed-{seed}.txt"
        best_round = rounds_dict[seed] - 1

        # Load CSV-style text file
        data = np.loadtxt(file_path, delimiter=",")
        if best_round >= data.shape[0]:
            raise ValueError(
                f"Round {best_round} out of range for {algo_name} seed {seed}"
            )

        accuracies = data[best_round, :]
        all_accuracies.extend(accuracies)

    # Plot combined distribution for each algorithm
    sns.histplot(
        all_accuracies,
        bins=25,
        kde=False,  # no smooth line
        label=f"{algo_name.upper()}",
        alpha=0.5,
    )

# --- Labels and layout ---
plt.title("Client Accuracy Distributions at Best Rounds (FEMNIST)")
plt.xlabel("Client Accuracy")
plt.ylabel("Number of Clients")
plt.legend()
plt.tight_layout()
plt.show()
