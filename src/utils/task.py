from collections import OrderedDict

import torch
import torch.nn.functional as F


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(
    net,
    trainloader,
    valloader,
    epochs,
    learning_rate,
    device,
    prox_mu=0.0,
    cli_strategy="fedavg",
    personal_net=None,
    local_epochs=1,
    q_ffl=0.0,  # 0 = standard FedAvg
    momentum=0.9,
):
    """Train the model on the training set with optional Q-FFL, FedProx, or Ditto."""

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Ditto: maintain personalized model copy and optimizer
    if cli_strategy == "ditto":
        global_params = [p.clone().detach() for p in net.parameters()]
        personal_net.to(device)
        personal_optimizer = torch.optim.SGD(
            personal_net.parameters(), lr=learning_rate, momentum=momentum
        )

    # FedProx global params
    if cli_strategy == "fedprox":
        global_params = [p.clone().detach() for p in net.parameters()]

    net.train()
    if cli_strategy == "ditto":
        personal_net.train()

    total_loss = 0
    total_personal_loss = 0

    for i in range(max(epochs, local_epochs)):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            if cli_strategy == "ditto":
                # --- Step 1: Train global model as usual (FedAvg) ---
                if i < epochs:
                    optimizer.zero_grad()
                    loss = criterion(net(images), labels)
                    # Q-FFL adjustment
                    if q_ffl > 0:
                        loss = (loss ** (q_ffl + 1)) / q_ffl
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # --- Step 2: Train personalized model with proximal term to global model ---
                if i < local_epochs:
                    personal_optimizer.zero_grad()
                    proximal_term = 0.0
                    for local_w, global_w in zip(
                        personal_net.parameters(), global_params
                    ):
                        proximal_term += (local_w - global_w).norm(2) ** 2

                    personal_loss = criterion(personal_net(images), labels)
                    if q_ffl > 0:
                        personal_loss = (personal_loss ** (q_ffl + 1)) / q_ffl
                    personal_loss += (prox_mu / 2) * proximal_term

                    personal_loss.backward()
                    personal_optimizer.step()
                    total_personal_loss += personal_loss.item()

            elif cli_strategy == "fedprox":
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                # Q-FFL adjustment
                if q_ffl > 0:
                    loss = (loss ** (q_ffl + 1)) / q_ffl
                proximal_term = 0.0
                for local_w, global_w in zip(net.parameters(), global_params):
                    proximal_term += (local_w - global_w).norm(2) ** 2
                loss += (prox_mu / 2) * proximal_term
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            else:
                # --- FedAvg (default) ---
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                # Q-FFL adjustment
                if q_ffl > 0:
                    loss = (loss ** (q_ffl + 1)) / q_ffl
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    train_loss = total_loss / len(trainloader)
    val_loss, val_acc = test(net, valloader, device)

    results = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader, device):
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
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
