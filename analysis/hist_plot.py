import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Configuration ---
# 0: FEMNIST, 1: CIFAR-10, 2: Shakespeare, 3: Fashion-MNIST, 4: CIFAR-10-CD
experiment = 0
seeds = [0, 1, 12345]
base_paths = [
    "./accuracies/femnist",
    "./accuracies/cifar10",
    "./accuracies/shakespeare",
    "./accuracies/fashion-mnist",
    "./accuracies/cifar10-cd",
]

# best rounds per algorithm
best_rounds_femnist = {
    "FedAvg": {0: 457, 1: 427, 12345: 465},
    "q-FedAvg": {0: 496, 1: 493, 12345: 494},
    "FedProx": {0: 390, 1: 377, 12345: 360},
    "FedYogi": {0: 452, 1: 500, 12345: 464},
    "Ditto": {0: 422, 1: 457, 12345: 452},
    "FedProxYogi": {0: 423, 1: 423, 12345: 439},
    "FedProxDitto": {0: 406, 1: 386, 12345: 465},
    "FedYogiDitto": {0: 407, 1: 484, 12345: 464},
    "FedProxYogiDitto": {0: 489, 1: 455, 12345: 474},
}

best_rounds_cifar = {
    "FedAvg": {0: 88, 1: 98, 12345: 99},
    "q-FedAvg": {0: 100, 1: 100, 12345: 100},
    "FedProx": {0: 95, 1: 95, 12345: 99},
    "FedYogi": {0: 81, 1: 86, 12345: 95},
    "Ditto": {0: 99, 1: 99, 12345: 100},
    "FedProxYogi": {0: 100, 1: 100, 12345: 99},
    "FedProxDitto": {0: 99, 1: 100, 12345: 98},
    "FedYogiDitto": {0: 90, 1: 96, 12345: 100},
    "FedProxYogiDitto": {0: 85, 1: 95, 12345: 91},
}

best_rounds_text = {
    "FedAvg": {0: 46, 1: 139, 12345: 65},
    "q-FedAvg": {0: 492, 1: 494, 12345: 500},
    "FedProx": {0: 312, 1: 134, 12345: 150},
    "FedYogi": {0: 95, 1: 241, 12345: 430},
    "Ditto": {0: 74, 1: 44, 12345: 101},
    "FedProxYogi": {0: 77, 1: 128, 12345: 436},
    "FedProxDitto": {0: 163, 1: 126, 12345: 59},
    "FedYogiDitto": {0: 450, 1: 173, 12345: 456},
    "FedProxYogiDitto": {0: 321, 1: 180, 12345: 370},
}

best_rounds_fashion = {
    "FedAvg": {0: 427, 1: 378, 12345: 489},
    "q-FedAvg": {0: 486, 1: 492, 12345: 494},
    "FedProx": {0: 312, 1: 134, 12345: 150},
    "FedYogi": {0: 372, 1: 495, 12345: 492},
    "Ditto": {0: 480, 1: 404, 12345: 354},
    "FedProxYogi": {0: 477, 1: 441, 12345: 355},
    "FedProxDitto": {0: 396, 1: 405, 12345: 371},
    "FedYogiDitto": {0: 236, 1: 157, 12345: 202},
    "FedProxYogiDitto": {0: 246, 1: 179, 12345: 212},
}

best_rounds_cifar_cd = {
    "FedAvg": {0: 448, 1: 494, 12345: 484},
    "q-FedAvg": {0: 498, 1: 468, 12345: 497},
    "FedProx": {0: 473, 1: 465, 12345: 497},
    "FedYogi": {0: 474, 1: 466, 12345: 476},
    "Ditto": {0: 493, 1: 487, 12345: 499},
    "FedProxYogi": {0: 332, 1: 490, 12345: 490},
    "FedProxDitto": {0: 499, 1: 500, 12345: 498},
    "FedYogiDitto": {0: 301, 1: 237, 12345: 285},
    "FedProxYogiDitto": {0: 233, 1: 260, 12345: 284},
}

best_rounds = [
    best_rounds_femnist,
    best_rounds_cifar,
    best_rounds_text,
    best_rounds_fashion,
    best_rounds_cifar_cd,
]

# --- Plot style ---
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))

# --- Load and plot both algorithms ---
for algo_name, rounds_dict in best_rounds[experiment].items():
    all_accuracies = []

    for seed in seeds:
        file_path = f"{base_paths[experiment]}/{algo_name.lower().replace('-', '')}-fashion-accs-seed-{seed}.txt"
        best_round = rounds_dict[seed] - 1
        print(file_path)
        # Load CSV-style text file
        data = np.loadtxt(file_path, delimiter=",")
        if best_round >= data.shape[0]:
            raise ValueError(
                f"Round {best_round} out of range for {algo_name} seed {seed}"
            )

        accuracies = data[best_round, :]
        all_accuracies.extend(accuracies)

    # Plot combined distribution for each algorithm
    kde = sns.kdeplot(
        all_accuracies,
        fill=True,
        label=algo_name,
        alpha=0.075,
        linewidth=1.5,
        bw_adjust=0.75,
        # stat="count",      # scale KDE to counts
        # common_norm=False, # don't normalize across multiple plots
    )

# --- Labels and layout ---
titles = [
    "FEMNIST",
    "CIFAR-10 - cross-silo",
    "Shakespeare",
    "Fashion-MNIST",
    "CIFAR-10 - cross-device",
]
plt.title(f"Client Accuracy Distributions at Best Rounds ({titles[experiment]})")
plt.xlabel("Client Accuracy")
plt.ylabel("Density")
plt.legend()
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
plt.tight_layout()
plt.show()
