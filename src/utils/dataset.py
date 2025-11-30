import random
import warnings

import torch
from datasets import concatenate_datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    NaturalIdPartitioner,
    PathologicalPartitioner,
    ShardPartitioner,
)
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from .language import letter_to_vec, word_to_indices

fds = None  # global cache
nid_to_cid = None  # mapping from partition index to character_id/writer_id for Shakespeare/FEMNIST
malicious_clients = None  # list of malicious client partition IDs


def prepare_shakespeare_fds(num_partitions: int = 31, seed: int = 42):
    """Prepares the Shakespeare dataset for experiments"""
    # Partition dataset by character (speaker)
    warnings.filterwarnings("ignore", category=UserWarning, module="flwr_datasets")

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
    """Prepares the FEMNIST dataset for experiments"""
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


def prepare_femnist_fds_skewed(
    num_partitions: int = 100,
    seed: int = 42,
    num_base_clients: int = 355,
    num_classes_per_partition: int = 5,
):
    """Prepares the FEMNIST dataset for experiments in SKEWED manner
    NOTE: This function requires a lot of memory since it materializes multiple writers' data!
    """

    # 1. Initialize Natural Partition (To select specific writers)
    base_fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
    )

    # 2. Get all available partitions and select the subsets of writers
    all_ids = range(base_fds.partitioners["train"].num_partitions)
    random.seed(seed)
    selected_ids = random.sample(all_ids, num_base_clients)

    # A. Load the actual data from the selected writers
    # We have to materialize the data here to reshuffle it
    writer_partitions = []
    for writer_node_id in selected_ids:
        # Load the specific writer's data
        writer_partitions.append(base_fds.load_partition(writer_node_id))

    # B. Merge into one large pool
    merged_dataset = concatenate_datasets(writer_partitions)

    # C. Apply PathologicalPartitioner to this pooled data
    # This ensures we only use data from the selected writers, but distribute
    # it so every client gets X classes.
    skewed_partitioner = PathologicalPartitioner(
        partition_by="character",
        num_partitions=num_partitions,
        num_classes_per_partition=num_classes_per_partition,
        seed=seed,
    )

    # Assign the merged dataset to the partitioner
    skewed_partitioner.dataset = merged_dataset

    # D. Return the partitioner as the 'fds' object
    # The partitioner object has a .load_partition() method, so it mimics fds behavior
    # The mapping is now 1-to-1 (0->0, 1->1) because we re-generated the partitions
    new_mapping = {i: i for i in range(num_partitions)}

    return skewed_partitioner, new_mapping


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset: str,
    seed: int = 42,
    hparam_tuning: bool = False,
    num_malicious_clients: int = 0,
    dir_alpha: float = 0.5,
    use_shards: bool = False,
):
    """Load partition (train/test) for CIFAR10, FEMNIST, or Shakespeare."""

    global fds
    global nid_to_cid
    global malicious_clients

    # --- CIFAR10 ---
    if dataset.lower() == "cifar10":
        if fds is None and not use_shards:
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={
                    "train": DirichletPartitioner(
                        partition_by="label",
                        num_partitions=num_partitions,
                        alpha=dir_alpha,
                        seed=seed,
                    ),
                    "test": DirichletPartitioner(
                        partition_by="label",
                        num_partitions=num_partitions,
                        alpha=dir_alpha,
                        seed=seed,
                    ),
                },
            )
        elif fds is None and use_shards:
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={
                    "train": ShardPartitioner(
                        partition_by="label",
                        num_partitions=num_partitions,
                        shard_size=100,
                        num_shards_per_partition=(50000 // 100) // num_partitions,
                        seed=seed,
                    ),
                },
            )

        transforms = Compose(
            [
                ToTensor(),
                Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),  # CIFAR10 normalization values
            ]
        )

        def apply_transforms(batch):
            batch["img"] = [transforms(img) for img in batch["img"]]
            return batch

        if hparam_tuning or use_shards:
            train_partition = fds.load_partition(partition_id, "train").with_transform(
                apply_transforms
            )
            # Divide data on each node: 80% train, 20% validation
            partition_train_test = train_partition.train_test_split(
                test_size=0.2, seed=seed
            )

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
        if fds is None and not use_shards:
            fds, nid_to_cid = prepare_femnist_fds(num_partitions, seed)
        elif fds is None and use_shards:
            fds, nid_to_cid = prepare_femnist_fds_skewed(
                num_partitions, seed, num_base_clients=355, num_classes_per_partition=5
            )

        if num_malicious_clients > 0 and malicious_clients is None:
            # Randomly select malicious clients
            malicious_clients = random.sample(
                range(num_partitions), num_malicious_clients
            )

        transforms = Compose(
            [ToTensor(), Normalize((0.1736,), (0.3317,))]
        )  # Based on EMNIST byclass data

        def apply_transforms(batch, is_malicious_client: bool = False):
            batch = {
                "img": [transforms(img) for img in batch["image"]],
                "label": batch["character"],
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
            if malicious_clients is not None:
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

        def apply_transforms(batch):
            return {
                "img": [torch.tensor(word_to_indices(data)) for data in batch["x"]],
                "label": [
                    torch.tensor(letter_to_vec(data)) for data in batch["y"]
                ],  # class index, not one-hot
            }

        partition = fds.load_partition(nid_to_cid[partition_id])
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
            partition_train_test = partition_train_test.with_transform(apply_transforms)
            trainloader = DataLoader(
                partition_train_test["train"],
                batch_size=batch_size,
                shuffle=True,
            )
            testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)

        return trainloader, testloader

    # --- Fashion-MNIST ---
    elif dataset.lower() == "fashion" or dataset.lower() == "fashion-mnist":
        if fds is None:
            fds = FederatedDataset(
                dataset="zalando-datasets/fashion_mnist",
                partitioners={
                    "train": ShardPartitioner(
                        partition_by="label",
                        num_partitions=num_partitions,
                        shard_size=60,
                        num_shards_per_partition=(60000 // 60) // num_partitions,
                        seed=seed,
                    ),
                },
            )

        def apply_transforms(batch):
            return {
                "img": [transforms(img) for img in batch["image"]],
                "label": batch["label"],
            }

        transforms = Compose(
            [ToTensor(), Normalize((0.2860,), (0.3530,))]  # Based on Fashion-MNIST data
        )
        partition = fds.load_partition(partition_id)
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
            partition_train_test = partition_train_test.with_transform(apply_transforms)
            trainloader = DataLoader(
                partition_train_test["train"],
                batch_size=batch_size,
                shuffle=True,
            )
            testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)

        return trainloader, testloader

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
