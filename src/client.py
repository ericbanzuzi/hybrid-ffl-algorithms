import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader

from src.models.cnn import CNN
from src.models.lstm import ShakespeareLSTM
from src.models.resnet import ResNet18
from src.utils.dataset import load_data
from src.utils.task import get_weights, set_weights, test, train


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        trainloader: DataLoader,
        valloader: DataLoader,
        local_epochs: int,
        learning_rate: float,
        strategy: str = "fedavg",
        dataset: str = "cifar10",
        net_type: str = "CNN",
        group_norm: bool = False,
    ):
        """Initialize the client with data loaders, hyperparameters, and model."""
        if net_type == "resnet18":
            self.net = ResNet18(dataset=dataset, BN_to_GN=group_norm)
        elif net_type == "rnn" or dataset in ["shakespeare"]:
            self.net = ShakespeareLSTM()
        else:
            self.net = CNN(dataset=dataset)

        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.strategy = strategy
        self.dataset = dataset

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        prox_mu = config.get("proximal_mu", 0.0)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
            prox_mu,
            self.strategy,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    dataset = context.run_config["dataset"]
    seed = context.run_config.get("seed", 42)
    hparam_tuning = context.run_config.get("hparam-tuning", 0) == 1
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    cli_strategy = context.run_config.get("cli-strategy", "fedavg")
    selected_model = context.run_config.get("model", "CNN").lower()
    group_norm = context.run_config.get("group-norm", 0) == 1

    trainloader, valloader = load_data(
        partition_id, num_partitions, batch_size, dataset, seed, hparam_tuning
    )

    # Return Client instance
    return FlowerClient(
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        cli_strategy,
        dataset,
        selected_model,
        group_norm,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
