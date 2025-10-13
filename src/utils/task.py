from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader


def get_weights(net: torch.nn.Module) -> list[np.ndarray]:
    """Return the weights of the model as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: torch.nn.Module, parameters: list[np.ndarray]):
    """Set the weights of the model from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(
    net,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device,
    prox_mu: float = 0.0,
    cli_strategy: str = "fedavg",
    personal_net=None,
    local_iterations: int = 1,
    qffl: bool = False,
    momentum: float = 0.0,
    local_learning_rate: float = 0.1,
    lam: float = 1.0,
):
    """Train the model on the training set with optional FedProx, or Ditto."""

    # For q-FFL we need to compute loss on the whole training data, with respect to the starting point (the global model)
    if qffl:
        local_global_loss, _ = test(net, trainloader, device)

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Ditto: maintain personalized model copy and optimizer
    if cli_strategy == "ditto":
        global_params = [p.clone().detach() for p in net.parameters()]
        personal_net.to(device)
        personal_optimizer = torch.optim.SGD(
            personal_net.parameters(), lr=local_learning_rate, momentum=momentum
        )

    # FedProx global params
    if cli_strategy == "fedprox":
        global_params = [p.clone().detach() for p in net.parameters()]

    net.train()
    if cli_strategy == "ditto":
        personal_net.train()

    total_loss = 0
    total_correct = 0
    total_personal_loss = 0
    for i in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            if cli_strategy == "fedprox":
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                proximal_term = 0.0
                for local_w, global_w in zip(net.parameters(), global_params):
                    proximal_term += (local_w - global_w).norm(2) ** 2
                loss += (prox_mu / 2) * proximal_term
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == labels).sum().item()
            else:
                # --- FedAvg (default) ---
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == labels).sum().item()

            if cli_strategy == "ditto":
                personal_optimizer.zero_grad()
                outputs = personal_net(images)
                loss = criterion(outputs, labels)

                # proximal regularization term: λ/2 * ||v_k - w_t||^2
                prox_term = 0.0
                for p_local, p_global in zip(personal_net.parameters(), global_params):
                    prox_term += torch.norm(p_local - p_global) ** 2
                loss += (lam / 2) * prox_term

                loss.backward()
                personal_optimizer.step()
                total_personal_loss += loss.item()

    train_loss = total_loss / len(trainloader)
    train_acc = total_correct / len(trainloader.dataset)
    val_loss, val_acc = test(net, valloader, device)

    if qffl:
        results = {
            "train-loss": train_loss,
            "train-accuracy": train_acc,
            "val-loss": val_loss,
            "val-accuracy": val_acc,
            "local-global-loss": local_global_loss,
        }
    else:
        results = {
            "train-loss": train_loss,
            "train-accuracy": train_acc,
            "val-loss": val_loss,
            "val-accuracy": val_acc,
        }
    return results


def test(net, testloader: DataLoader, device):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
