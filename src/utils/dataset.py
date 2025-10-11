import random

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, NaturalIdPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None  # global cache
nid_to_cid = None  # mapping from partition index to character_id/writer_id for Shakespeare/FEMNIST
malicious_clients = None  # list of malicious client partition IDs


def prepare_shakespeare_fds(num_partitions: int = 31, seed: int = 42):
    # Partition dataset by character (speaker)
    base_fds = FederatedDataset(
        dataset="flwrlabs/shakespeare",
        partitioners={"train": NaturalIdPartitioner(partition_by="character_id")},
    )

    # Get all available partitions
    all_ids = range(base_fds.partitioners["train"].num_partitions)

    # Return a *view* of the dataset restricted to selected roles
    random.seed(seed)
    selected_ids = random.sample(all_ids, num_partitions)

    return base_fds, {i: cid for i, cid in enumerate(selected_ids)}


def prepare_femnist_fds(num_partitions: int = 500, seed: int = 42):
    # Partition dataset by writer (writer_id)
    base_fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
    )

    # Get all available partitions
    all_ids = range(base_fds.partitioners["train"].num_partitions)

    # Return a *view* of the dataset restricted to selected roles
    random.seed(seed)
    selected_ids = random.sample(all_ids, num_partitions)

    return base_fds, {i: cid for i, cid in enumerate(selected_ids)}


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset: str,
    seed: int = 42,
    hparam_tuning: bool = False,
    num_malicious_clients: int = 0,
):
    """Load partition (train/test) for CIFAR10, FEMNIST, or Shakespeare."""

    global fds
    global nid_to_cid
    global malicious_clients

    # --- CIFAR10 ---
    if dataset.lower() == "cifar10":
        if fds is None:
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={
                    "train": DirichletPartitioner(
                        partition_by="label",
                        num_partitions=num_partitions,
                        alpha=0.5,
                        seed=seed,
                    ),
                    "test": DirichletPartitioner(
                        partition_by="label",
                        num_partitions=num_partitions,
                        alpha=0.5,
                        seed=seed,
                    ),
                },
            )

        transforms = Compose(
            [
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        def apply_transforms(batch):
            batch["img"] = [transforms(img) for img in batch["img"]]
            return batch

        if hparam_tuning:
            train_partition = fds.load_partition(partition_id, "train").with_transform(
                apply_transforms
            )
            # Divide data on each node: 80% train, 20% validation
            partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

            trainloader = DataLoader(
                partition_train_test["train"], batch_size=batch_size, shuffle=True
            )
            testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        else:
            train_partition = fds.load_partition(partition_id, "train").with_transform(
                apply_transforms
            )
            test_partition = fds.load_partition(partition_id, "test").with_transform(
                apply_transforms
            )

            trainloader = DataLoader(
                train_partition, batch_size=batch_size, shuffle=True
            )
            testloader = DataLoader(test_partition, batch_size=batch_size)

        return trainloader, testloader

    # --- FEMNIST ---
    elif dataset.lower() == "femnist" or dataset.lower() == "mnist":
        if fds is None:
            fds, nid_to_cid = prepare_femnist_fds(num_partitions, seed)

        if num_malicious_clients > 0 and malicious_clients is None:
            # Randomly select malicious clients
            malicious_clients = random.sample(
                range(num_partitions), num_malicious_clients
            )

        transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

        def apply_transforms(batch, is_malicious_client: bool = False):
            batch = {
                "img": [transforms(img) for img in batch["image"]],
                "label": [y for y in batch["character"]],
            }

            # Flip labels randomly (label poisoning)
            if is_malicious_client:
                num_classes = 62
                random.seed(
                    seed + partition_id
                )  # ensures same labels flipped each time
                batch["label"] = [
                    random.randint(0, num_classes - 1) for _ in batch["label"]
                ]
            return batch

        partition = fds.load_partition(nid_to_cid[partition_id])
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

        if hparam_tuning:
            partition_train = partition_train_test["train"].with_transform(
                apply_transforms
            )
            # Divide data on each node: 80% train, 20% validation
            partition_train_val = partition_train.train_test_split(
                test_size=0.2, seed=seed
            )

            trainloader = DataLoader(
                partition_train_val["train"], batch_size=batch_size, shuffle=True
            )
            testloader = DataLoader(partition_train_val["test"], batch_size=batch_size)
        else:
            if not malicious_clients is None:
                is_malicious_client = partition_id in malicious_clients
                partition_train = partition_train_test["train"].with_transform(
                    apply_transforms(is_malicious_client=is_malicious_client)
                )
                partition_test = partition_train_test["test"].with_transform(
                    apply_transforms
                )
                trainloader = DataLoader(
                    partition_train, batch_size=batch_size, shuffle=True
                )
                testloader = DataLoader(partition_test, batch_size=batch_size)
            else:
                partition_train_test = partition_train_test.with_transform(
                    apply_transforms
                )
                trainloader = DataLoader(
                    partition_train_test["train"], batch_size=batch_size, shuffle=True
                )
                testloader = DataLoader(
                    partition_train_test["test"], batch_size=batch_size
                )
        return trainloader, testloader

    # --- Shakespeare ---
    elif dataset.lower() == "shakespeare":
        if fds is None:
            fds, nid_to_cid = prepare_shakespeare_fds(num_partitions, seed)

        partition = fds.load_partition(nid_to_cid[partition_id])
        partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

        def collate_fn(batch):
            # TODO: appropriate padding/truncation and tokenization
            x = torch.tensor([b["x"] for b in batch], dtype=torch.long)
            y = torch.tensor([b["y"] for b in batch], dtype=torch.long)
            return {"x": x, "y": y}

        trainloader = DataLoader(
            partition_train_test["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        testloader = DataLoader(
            partition_train_test["test"], batch_size=batch_size, collate_fn=collate_fn
        )
        return trainloader, testloader

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
