"""pytorchexample: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx

from src.utils.task import get_weights
from src.models.cnn import CNN
from src.strategy.adafed import AdaFedStrategy


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    dataset = context.run_config["dataset"]
    
    # Initialize model parameters
    ndarrays = get_weights(CNN(dataset=dataset))
    parameters = ndarrays_to_parameters(ndarrays)

    base_kwargs = {
        "fraction_fit": context.run_config["fraction-fit"],
        "fraction_evaluate": context.run_config["fraction-evaluate"],
        "min_available_clients": 2,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "initial_parameters": parameters,
    }
    
    # Define the strategy
    if context.run_config["cli-strategy"] == "fedprox":
        strategy = FedProx(
            **base_kwargs,
            proximal_mu=context.run_config["proximal-mu"]
        )
    else:
        strategy = FedAvg(**base_kwargs)
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)