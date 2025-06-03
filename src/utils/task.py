from collections import OrderedDict

import torch
import torch.nn.functional as F


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, valloader, epochs, learning_rate, device, prox_mu, cli_strategy = 'fedavg'):
    """Train the model on the training set."""
    if cli_strategy == 'fedprox':
        global_params = [p.clone().detach() for p in net.parameters()]
    else:
        pass

    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()

    total_loss = 0, 
    for _ in range(epochs):
        total_loss = 0
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()

            if cli_strategy == 'fedprox':
                proximal_term = 0.0
                for local_weights, global_weights in zip(net.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)
                loss = criterion(net(images.to(device)), labels) + (prox_mu / 2) * proximal_term
            else:
                loss = criterion(net(images.to(device)), labels.to(device))
            
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
