"""pytorchexample: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx

from src.utils.task import get_weights
from src.models.cnn import CNN
from src.models.lstm import ShakespeareLSTM
from src.strategy.adafed import AdaFedStrategy


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
    
    # Initialize model parameters
    model = CNN(dataset=dataset) if model_type == 'cnn' else ShakespeareLSTM()
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
            **base_kwargs,
            proximal_mu=context.run_config["proximal-mu"],
        )
    else:
        strategy = FedAvg(**base_kwargs)
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)