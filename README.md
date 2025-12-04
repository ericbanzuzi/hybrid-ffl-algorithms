# HybridFFL: Fair Federated Learning via Client Drift Reduction, Adaptive Optimization and Personalization

This repository contains the implementation of **HybridFFL**, a Federated Learning (FL) framework designed to improve fairness, scalability, and robustness in distributed neural network training by combining orthogonal FL methods. 


## 🚀 Overview

**HybridFFL** investigates the orthogonal combination of three state-of-the-art FL techniques to address heterogeneity in cross-device settings:
1.  **Client Drift Reduction:** Uses **FedProx** to stabilize local updates via proximal regularization.
2.  **Adaptive Optimization:** Uses **FedYogi** for server-side adaptive learning rates.
3.  **Personalization:** Uses **Ditto** to train personalized local models alongside the global model.

The framework supports multiple hybrid variations:
* `FedProxYogi`
* `FedProxDitto`
* `FedYogiDitto`
* `FedProxYogiDitto` (Full Hybrid)


## 📊 Datasets

The repository includes setups for four benchmark datasets, covering both image classification and text generation:

| Dataset | Type | Heterogeneity |
| :--- | :--- | :--- |
| **FEMNIST** | Image | Natural (Writer ID) |
| **Fashion-MNIST** | Image | Pathological (Shards) |
| **CIFAR-10** | Image | Dirichlet / Shards |
| **Shakespeare** | Text | Natural (Speaking Roles) |

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ericbanzuzi/hybrid-ffl-algorithms.git
    cd hybrid-ffl-algorithms
    ```

2.  **Install dependencies:**
    Install the dependencies defined in `pyproject.toml`:

    ```bash
    pip install -e .
    ```

    Or alternatively:

    ```bash
    pip install -r requirements.txt
    ```

## Project structure

The content of the project is structured as follows:
```shell
hybrid-fl-algorithms
├── analysis
├── experiments
├── notebooks
├── src
│   ├── __init__.py
│   ├── client.py   # Defines the ClientApp
│   ├── server.py   # Defines the ServerApp
│   ├── models
│   │   ├── lstm.py
│   │   └── cnn.py
│   ├── strategy    # Contains all aggregation strategies for servers
│   │   ├── fedavg.py
│   │   ├── fedprox.py
│   │   ├── fedyogi.py
│   │   ├── qfedavg.py
│   │   └── adafed.py
│   └── utils
│   │   ├── dataset.py      # Defines functions for data loading
│   │   ├── language.py      # Defines functions for text data preprocessing
│       └── task.py         # Defines functions for training
├── pyproject.toml      # Project metadata like dependencies and configs
├── Makefile      # Useful commands for running things related to the project
└── README.md
```

## 🏃 Usage

You can run simulations using the main script. The system uses **Weights & Biases** for experiment tracking.

### Basic Run with the Simulation Engine

There exists multiple predefined simulation engine confugrations in `pyproject.toml`, which can be used to run simulations. To make custom setups with different simulations you can create more in a similar fashion.

To run the full **HybridFFL** (FedProx + Yogi + Ditto) on **FEMNIST**:

```bash
flwr run . femnist-sim \
    --run-config '\
        num-server-rounds=500 \
        agg-strategy="fedyogi" \
        cli-strategy="fedproxditto" \
        dataset="femnist" \
        model="cnn" \
        batch-size=20 \
        agg-learning-rate=0.00316  \
        learning-rate=0.1 \
        proximal-mu=0.1 \
        lambda=1 \
        fraction-fit=0.03 \
        fraction-evaluate=1 \
        store-client-accs=1 \
        client-acc-file="femnist/fedproxyogiditto-femnist-accs"'
```