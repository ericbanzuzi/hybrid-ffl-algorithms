import json
import os
from logging import INFO, WARNING
from typing import Optional, Union

import numpy as np
import torch
from flwr.common import (
    ArrayRecord,
    EvaluateRes,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    logger,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import QFedAvg
from flwr.server.strategy.aggregate import aggregate_qffl

import wandb


# pylint: disable=too-many-locals
class CustomQFedAvg(QFedAvg):
    """
    Configurable QFedAvg strategy implementation.
    A strategy that keeps the core functionality of QFedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases.
    """

    def __init__(
        self,
        model_type: str = "cnn",
        dataset: str = "femnist",
        seed: int = 42,
        num_rounds: int = None,
        save_model: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}

        # Log those same metrics to W&B
        project = f"qfedavg-{dataset}-{model_type}"
        wandb.init(project=project, name=f"qfedavg-seed-{seed}")
        self.model_type = model_type
        self.dataset = dataset
        self.best_acc_so_far = 0.0  # Track best accuracy to save model checkpoints
        self.seed = seed
        self.num_rounds = num_rounds
        self.save_model = save_model

        self.results_dir = f"./experiment-results/{dataset}-{model_type}/"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.checkpoint_dir = f"./checkpoints/{dataset}-{model_type}/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

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
            file_name = f"qfedavg-round-{current_round}-seed-{self.seed}.pt"
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
        # Convert results

        def norm_grad(grad_list: NDArrays) -> float:
            # input: nested gradients
            # output: square of the L-2 norm
            client_grads = grad_list[0]
            for i in range(1, len(grad_list)):
                client_grads = np.append(
                    client_grads, grad_list[i]
                )  # output a flattened array
            squared = np.square(client_grads)
            summed = np.sum(squared)
            return float(summed)

        deltas = []
        hs_ffl = []

        if self.pre_weights is None:
            raise AttributeError("QffedAvg pre_weights are None in aggregate_fit")

        weights_before = self.pre_weights

        for _, fit_res in results:
            new_weights = parameters_to_ndarrays(fit_res.parameters)
            loss = fit_res.metrics["local-global-loss"]
            # plug in the weight updates into the gradient
            grads = [
                np.multiply((u - v), 1.0 / self.learning_rate)
                for u, v in zip(weights_before, new_weights)
            ]
            deltas.append(
                [np.float_power(loss + 1e-10, self.q_param) * grad for grad in grads]
            )
            # estimation of the local Lipschitz constant
            hs_ffl.append(
                self.q_param
                * np.float_power(loss + 1e-10, (self.q_param - 1))
                * norm_grad(grads)
                + (1.0 / self.learning_rate)
                * np.float_power(loss + 1e-10, self.q_param)
            )

        weights_aggregated: NDArrays = aggregate_qffl(weights_before, deltas, hs_ffl)
        parameters_aggregated = ndarrays_to_parameters(weights_aggregated)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Store metrics as dictionary
        my_results = metrics_aggregated
        # Insert into local dictionary
        self.results_to_save[server_round] = my_results

        # Save metrics as json
        with open(
            f"{self.results_dir}/qfedavg-results-seed-{self.seed}.json", "w"
        ) as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Save new Global Model as a PyTorch checkpoint
        self._update_best_acc(
            current_round=server_round,
            accuracy=metrics_aggregated.get("avg-val-accuracy", 0.0),
            arrays=ArrayRecord.from_numpy_ndarrays(weights_aggregated),
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
            f"{self.results_dir}/qfedavg-results-seed-{self.seed}.json", "w"
        ) as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Return the expected outputs for `evaluate`
        return loss_aggregated, metrics_aggregated
