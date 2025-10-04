"""pytorchexample: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAdam, FedAvg, FedProx, FedYogi

from src.models.cnn import CNN, ResNet18
from src.models.lstm import ShakespeareLSTM
from src.strategy.adafed import AdaFedStrategy
from src.utils.task import get_weights


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * metric["accuracy"] for num_examples, metric in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    dataset = context.run_config["dataset"]
    agg_strategy = context.run_config.get("agg-strategy", "fedavg")
    model_type = context.run_config.get("model", "CNN").lower()
    use_yogi = context.run_config.get("yogi-server", 0) == 1
    use_adam = context.run_config.get("adam-server", 0) == 1

    # Initialize model parameters
    if model_type == "resnet18":
        model = ResNet18(dataset=dataset)
    elif model_type == "cnn" or dataset in ["shakespeare"]:
        model = ShakespeareLSTM()
    else:
        model = CNN(dataset=dataset)

    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    base_kwargs = {
        "fraction_fit": context.run_config["fraction-fit"],
        "fraction_evaluate": context.run_config["fraction-evaluate"],
        "min_available_clients": 2,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "initial_parameters": parameters,
    }

    # Define the strategy
    if agg_strategy == "fedprox":
        strategy = FedProx(
            **base_kwargs,
            proximal_mu=context.run_config["proximal-mu"],
        )
    elif agg_strategy == "adafed":
        strategy = AdaFedStrategy(
            use_yogi=use_yogi,
            use_adam=use_adam,
            **base_kwargs,
            proximal_mu=context.run_config["proximal-mu"],
        )
    elif agg_strategy == "fedadam":
        strategy = FedAdam(**base_kwargs)
    elif agg_strategy == "fedyogi":
        strategy = FedYogi(**base_kwargs)
    else:
        strategy = FedYogi(**base_kwargs)

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
