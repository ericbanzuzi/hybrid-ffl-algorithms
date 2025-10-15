import warnings
from typing import List, Tuple

import numpy as np
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.models.cnn import CNN
from src.models.lstm import ShakespeareLSTM
from src.models.resnet import ResNet18
from src.strategy.adafed import AdaFed
from src.strategy.fedavg import CustomFedAvg
from src.strategy.fedprox import CustomFedProx
from src.strategy.fedyogi import CustomFedYogi
from src.strategy.qfedavg import CustomQFedAvg
from src.utils.task import get_weights

warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb")


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * metric["accuracy"] for num_examples, metric in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def on_fit_config(server_round: int) -> dict:
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 1 if server_round < 2 else 2,
        "proximal_mu": 0.0,
    }
    return config


def aggregate_eval_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregates client metrics to compute:
    - Mean accuracy (unweighted)
    - Standard deviation of accuracies (unweighted)
    - Worst and best 10% accuracies (unweighted)
    - Mean loss (unweighted)
    """
    accuracies = np.array([metric["accuracy"] for _, metric in metrics])

    # Mean accuracy and standard deviation (unweighted, across clients)
    mean_acc = float(np.mean(accuracies))
    std = float(np.std(accuracies, ddof=0))

    # Worst 10% and best 10% (you can adjust to 5% if needed)
    sorted_acc = np.sort(accuracies)
    k = len(sorted_acc)
    n10 = max(1, int(0.1 * k))
    worst_10_mean = float(np.mean(sorted_acc[:n10]))
    best_10_mean = float(np.mean(sorted_acc[-n10:]))

    return {
        "avg-test-accuracy": mean_acc,
        "std-test-accuracy": std,
        "worst-10": worst_10_mean,
        "best-10": best_10_mean,
    }


def aggregate_fit_metrics(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """
    Aggregates client metrics to compute:
    - Mean accuracy (unweighted)
    - Standard deviation of accuracies (unweighted)
    - Worst and best 10% accuracies (unweighted)
    - Mean loss (unweighted)
    """
    train_accuracies = np.array([metric["train-accuracy"] for _, metric in metrics])
    train_losses = np.array([metric.get("train-loss", 0.0) for _, metric in metrics])

    # Mean accuracy, standard deviation, mean loss (unweighted, across clients)
    mean_acc_train = float(np.mean(train_accuracies))
    std_acc_train = float(np.std(train_accuracies, ddof=0))
    mean_loss_train = float(np.mean(train_losses))

    val_accuracies = np.array([metric["val-accuracy"] for _, metric in metrics])
    val_losses = np.array([metric.get("val-loss", 0.0) for _, metric in metrics])

    # Mean accuracy, standard deviation, mean loss (unweighted, across clients)
    mean_acc_val = float(np.mean(val_accuracies))
    std_val = float(np.std(val_accuracies, ddof=0))
    mean_loss_val = float(np.mean(val_losses))

    return {
        "avg-train-accuracy": mean_acc_train,
        "std-train-accuracy": std_acc_train,
        "avg-train-loss": mean_loss_train,
        "avg-val-accuracy": mean_acc_val,
        "std-val-accuracy": std_val,
        "avg-val-loss": mean_loss_val,
    }


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config.get("num-server-rounds", 10)
    dataset = context.run_config.get("dataset", "femnist").lower()
    agg_strategy = context.run_config.get("agg-strategy", "fedavg").lower()
    selected_model = context.run_config.get("model", "CNN").lower()
    use_yogi = context.run_config.get("yogi-server", 0) == 1
    use_adam = context.run_config.get("adam-server", 0) == 1
    group_norm = context.run_config.get("group-norm", 0) == 1
    seed = context.run_config.get("seed", 42)
    proximal_mu = context.run_config.get("proximal-mu", 0.0)
    lr = context.run_config.get("agg-learning-rate") or context.run_config.get(
        "learning-rate", 0.1
    )
    cli_strategy = context.run_config.get("cli-strategy", "fedavg").lower()
    gamma = context.run_config.get("gamma", 1.0)
    tau = context.run_config.get("lambda", 1.0)

    # Initialize model parameters
    if selected_model == "resnet18":
        model = ResNet18(dataset=dataset, BN_to_GN=group_norm)
    elif selected_model in ["rnn", "lstm"] or dataset in ["shakespeare"]:
        model = ShakespeareLSTM()
    else:
        model = CNN(dataset=dataset)

    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    base_kwargs = {
        "fraction_fit": context.run_config["fraction-fit"],
        "fraction_evaluate": context.run_config["fraction-evaluate"],
        "min_available_clients": context.run_config.get("min-clients", 2),
        "evaluate_metrics_aggregation_fn": aggregate_eval_metrics,
        "fit_metrics_aggregation_fn": aggregate_fit_metrics,
        "initial_parameters": parameters,
    }

    # Define the strategy
    if agg_strategy == "fedprox":
        strategy = CustomFedProx(
            **base_kwargs,
            model_type=selected_model,
            dataset=dataset,
            seed=seed,
            proximal_mu=proximal_mu,
        )
    elif agg_strategy == "adafed":
        strategy = AdaFed(
            **base_kwargs,
            use_yogi=use_yogi,
            use_adam=use_adam,
            proximal_mu=proximal_mu,
            cli_strategy=cli_strategy,
            model_type=selected_model,
            dataset=dataset,
            seed=seed,
            lr=lr,
            gamma=gamma,
            tau=tau,
        )
    elif agg_strategy in ["fedyogi", "yogi"]:
        strategy = CustomFedYogi(
            **base_kwargs,
            proximal_mu=proximal_mu,
            cli_strategy=cli_strategy,
            model_type=selected_model,
            dataset=dataset,
            seed=seed,
        )
    elif agg_strategy in ["qfedavg", "qffl"]:
        strategy = CustomQFedAvg(
            **base_kwargs,
            model_type=selected_model,
            dataset=dataset,
            seed=seed,
        )
    else:
        strategy = CustomFedAvg(
            **base_kwargs,
            model_type=selected_model,
            dataset=dataset,
            seed=seed,
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
