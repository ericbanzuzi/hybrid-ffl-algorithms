import json
import os
from datetime import datetime
from logging import INFO, WARNING
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from flwr.common import (
    ArrayRecord,
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    logger,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace

import wandb


class CustomFedProx(FedProx):
    """A strategy that keeps the core functionality of FedProx unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases.
    """

    def __init__(
        self,
        model_type: str = "cnn",
        dataset: str = "cifar10",
        seed: int = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}

        # Log those same metrics to W&B
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        project = f"fedprox-{dataset}-{model_type}"
        wandb.init(project=project, name=f"fedprox-seed-{seed}")
        self.model_type = model_type
        self.dataset = dataset
        self.best_acc_so_far = 0.0  # Track best accuracy to save model checkpoints
        self.seed = seed

        self.results_dir = f"./experiment-results/{dataset}-{model_type}/"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.checkpoint_dir = f"./checkpoints/{dataset}-{model_type}/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        """Set the path where results and model checkpoints will be saved."""
        self.save_path = path
        self.run_dir = run_dir

    def _update_best_acc(
        self, current_round: int, accuracy: float, arrays: ArrayRecord
    ) -> None:
        """Update best accuracy and save model checkpoint if current accuracy is higher."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # Save the PyTorch model
            file_name = f"fedprox-round-{current_round}-seed-{self.seed}.pt"
            torch.save(
                arrays.to_torch_state_dict(), f"{self.checkpoint_dir}/{file_name}"
            )
            logger.log(INFO, "💾 New best model saved to disk: %s", file_name)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Store metrics as dictionary
        my_results = metrics_aggregated
        # Insert into local dictionary
        self.results_to_save[server_round] = my_results

        # Save metrics as json
        with open(
            f"{self.results_dir}/fedprox-results-seed-{self.seed}.json", "w"
        ) as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Save new Global Model as a PyTorch checkpoint
        self._update_best_acc(
            current_round=server_round,
            accuracy=metrics_aggregated.get("avg-val-accuracy", 0.0),
            arrays=ArrayRecord.from_numpy_ndarrays(aggregated_ndarrays),
        )

        # Return the expected outputs for `fit`
        return parameters_aggregated, metrics_aggregated

    def loss_avg(
        self,
        losses: list[float],
    ) -> tuple[float, float]:
        """Compute weighted average loss."""
        if not losses:
            return 0.0, 0.0
        return np.mean(losses), np.std(losses, ddof=0)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        # Modified from FedAvg.aggregate_evaluate to save metrics and log to W&B
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated, std_loss_aggregated = self.loss_avg(
            [evaluate_res.loss for _, evaluate_res in results]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        # Store metrics as dictionary
        my_results = {
            "avg-test-loss": loss_aggregated,
            "std-test-loss": std_loss_aggregated,
            **metrics_aggregated,
        }

        # Insert into local dictionary
        self.results_to_save[server_round] = {
            **self.results_to_save[server_round],
            **my_results,
        }

        # Save metrics as json
        with open(
            f"{self.results_dir}/fedprox-results-seed-{self.seed}.json", "w"
        ) as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Return the expected outputs for `evaluate`
        return loss_aggregated, metrics_aggregated
