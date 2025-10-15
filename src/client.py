import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, Context
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
        group_norm: bool = True,
        proximal_mu: float = 0.0,
        qffl: bool = False,
        lam: float = 1.0,
    ):
        """Initialize the client with data loaders, hyperparameters, and model."""
        if net_type == "resnet18":
            self.net = ResNet18(dataset=dataset, BN_to_GN=group_norm)
        elif net_type in ["rnn", "lstm"] or dataset in ["shakespeare"]:
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
        self.proximal_mu = proximal_mu
        self.qffl = qffl
        self.personal_net = None
        self.lam = lam

        if (
            strategy == "ditto"
            and "personal_net" not in self.client_state.config_records
        ):
            if net_type == "resnet18":
                self.personal_net = ResNet18(dataset=dataset, BN_to_GN=group_norm)
            elif net_type == "rnn" or dataset in ["shakespeare"]:
                self.personal_net = ShakespeareLSTM()
            else:
                self.personal_net = CNN(dataset=dataset)

            self.client_state.config_records["personal_net"] = ArrayRecord(
                get_weights(self.personal_net)
            )

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)

        if self.strategy == "ditto":
            set_weights(
                self.personal_net, self.client_state.config_records["personal_net"]
            )

        results = train(
            net=self.net,
            trainloader=self.trainloader,
            valloader=self.valloader,
            epochs=self.local_epochs,
            learning_rate=self.lr,
            device=self.device,
            prox_mu=self.proximal_mu,
            cli_strategy=self.strategy,
            qffl=self.qffl,
            personal_net=self.personal_net,
        )

        if self.strategy == "ditto":
            # after updating personal_net inside train()
            self.client_state.config_records["personal_net"] = ArrayRecord(
                get_weights(self.personal_net)
            )

        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        if self.strategy == "ditto":
            set_weights(self.net, self.client_state.config_records["personal_net"])
        else:
            set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config[
        "partition-id"
    ]  # 0, 1, 2, ..., K -1, client id assigned to this node
    num_partitions = context.node_config["num-partitions"]  # K, total number of clients

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config.get("batch-size", 32)
    dataset = context.run_config.get("dataset", "femnist").lower()
    seed = context.run_config.get("seed", 42)
    hparam_tuning = context.run_config.get("hparam-tuning", 0) == 1
    local_epochs = context.run_config.get("local-epochs", 1)
    learning_rate = context.run_config.get("learning-rate", 0.1)
    cli_strategy = context.run_config.get("cli-strategy", "fedavg")
    selected_model = context.run_config.get("model", "CNN").lower()
    group_norm = context.run_config.get("group-norm", 0) == 1
    proximal_mu = context.run_config.get("proximal-mu", 0.0)
    qffl = context.run_config.get("agg-strategy") in ["qffl", "qfedavg"]
    lam = context.run_config.get("lambda", 1)

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
        proximal_mu,
        qffl,
        lam,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
