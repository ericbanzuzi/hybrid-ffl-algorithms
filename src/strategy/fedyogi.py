import json
import os
from logging import INFO, WARNING
from typing import Callable, Optional, Union

import numpy as np
import torch
from flwr.common import (
    ArrayRecord,
    EvaluateRes,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    logger,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedOpt

import wandb


class CustomFedYogi(FedOpt):
    """
    A strategy that keeps the core functionality of FedYogi unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases.
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        model_type: str = "cnn",
        dataset: str = "femnist",
        seed: int = 42,
        proximal_mu: float = 0.0,
        cli_strategy: str = "fedavg",
        num_rounds: int = None,
        save_model: bool = False,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}

        # Log those same metrics to W&B
        project = f"fedyogi-{cli_strategy}-{dataset}-{model_type}"
        wandb.init(project=project, name=f"fedyogi-{cli_strategy}-seed-{seed}")
        self.model_type = model_type
        self.dataset = dataset
        self.best_acc_so_far = 0.0  # Track best accuracy to save model checkpoints
        self.seed = seed
        self.proximal_mu = proximal_mu
        self.cli_strategy = cli_strategy
        self.num_rounds = num_rounds
        self.save_model = save_model

        self.results_dir = f"./experiment-results/{dataset}-{model_type}/"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.checkpoint_dir = f"./checkpoints/{dataset}-{model_type}/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedYogi(accept_failures={self.accept_failures})"
        return rep

    def _update_best_acc(
        self, current_round: int, accuracy: float, arrays: ArrayRecord
    ) -> None:
        """Update best accuracy and save model checkpoint if current accuracy is higher."""
        if self.save_model and (
            accuracy > self.best_acc_so_far or current_round == self.num_rounds
        ):
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # Save the PyTorch model
            file_name = (
                f"fedyogi-{self.cli_strategy}-round-{current_round}-seed-{self.seed}.pt"
            )
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
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        # Yogi
        delta_t: NDArrays = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            x - (1.0 - self.beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights

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
            f"{self.results_dir}/fedyogi-{self.cli_strategy}-results-seed-{self.seed}.json",
            "w",
        ) as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Save new Global Model as a PyTorch checkpoint
        self._update_best_acc(
            current_round=server_round,
            accuracy=metrics_aggregated.get("avg-val-accuracy", 0.0),
            arrays=ArrayRecord.from_numpy_ndarrays(self.current_weights),
        )

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated

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
            f"{self.results_dir}/fedyogi-{self.cli_strategy}-results-seed-{self.seed}.json",
            "w",
        ) as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Return the expected outputs for `evaluate`
        return loss_aggregated, metrics_aggregated
