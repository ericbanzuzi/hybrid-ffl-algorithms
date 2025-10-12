import json
import os
from logging import INFO, WARNING
from typing import List, Optional, Union

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
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import wandb


class AdaFedStrategy(FedAvg):
    """Custom AdaFed strategy computing pseudo-gradients.

    Implementation based on "AdaFed: Fair Federated Learning via Adaptive Common Descent Direction"
    from https://arxiv.org/abs/2401.04993.
    """

    def __init__(
        self,
        model_type: str = "cnn",
        dataset: str = "femnist",
        seed: int = 42,
        cli_strategy: str = "fedavg",
        gamma: float = 1.0,
        lr: float = 0.1,
        use_yogi: bool = False,
        use_adam: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.99,
        m_t: Optional[np.ndarray] = None,
        v_t: Optional[np.ndarray] = None,
        tau: float = 1e-10,  # Small constant for stability
        proximal_mu: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.global_weights: Optional[List[np.ndarray]] = None
        self.lr = lr
        self.use_yogi = use_yogi
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_t = m_t  # type: ignore
        self.v_t = v_t  # type: ignore
        self.tau = tau
        self.t = 0  # Time step for adaptive optimizers
        self.cli_strategy = cli_strategy
        self.proximal_mu = proximal_mu

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}

        # Log those same metrics to W&B
        project = f"adafed-{cli_strategy}-{dataset}-{model_type}"
        wandb.init(project=project, name=f"adafed-{cli_strategy}-seed-{seed}")
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

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"AdaFed(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def _update_best_acc(
        self, current_round: int, accuracy: float, arrays: ArrayRecord
    ) -> None:
        """Update best accuracy and save model checkpoint if current accuracy is higher."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # Save the PyTorch model
            file_name = (
                f"adafed-{self.cli_strategy}-round-{current_round}-seed-{self.seed}.pt"
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
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        # Assume self.initial_parameters is already a list of np.ndarrays
        assert (
            self.initial_parameters is not None
        ), "When using server-side optimization, model needs to be initialized."

        initial_parameters = parameters_to_ndarrays(self.initial_parameters)
        gradients_list, losses = [], []
        for _, fit_res in results:
            # Compute pseudo-gradient as single d-dimensional vectors
            pseudo_gradient: NDArrays = np.concatenate(
                [
                    x - y
                    for x, y in zip(
                        initial_parameters, parameters_to_ndarrays(fit_res.parameters)
                    )
                ],
                axis=None,
            )

            gradients_list.append(pseudo_gradient)
            losses.append(fit_res.metrics["train-loss"])

        gradients_matrix = np.stack(gradients_list)

        # Orthogonalize gradients
        orthograds = self.modified_gram_schmidt(gradients_matrix, losses, self.gamma)

        # Minimum norm element
        d_t = self.compute_convex_combination(orthograds)
        logger.log(INFO, f"🚥 Computed d_t with norm {np.linalg.norm(d_t)}")

        # fedavg
        # client_examples = [res.num_examples for _, res in results]

        # d_t = np.sum([
        #     (num_examples / sum(client_examples)) * grad
        #     for num_examples, grad in zip(client_examples, gradients_list)
        # ], axis=0)

        # Debugging info
        # for i in range(min(3, len(orthograds))):
        #     for j in range(i+1, min(3, len(orthograds))):
        #         cosine = np.dot(orthograds[i], orthograds[j]) / (np.linalg.norm(orthograds[i]) * np.linalg.norm(orthograds[j]))
        #         print(f"Cosine between orthograd {i} and {j}: {cosine}")

        # print(f"Gradient shape: {gradients_matrix.shape}")
        # print(f"d_t norm: {np.linalg.norm(d_t)}")
        # print(f"Initial param norm: {np.linalg.norm(np.concatenate([p.flatten() for p in initial_parameters]))}")

        if self.use_yogi or self.use_adam:
            d_t = self.compute_adjusted_dt(d_t, initial_parameters)

        # Update current weights
        adafed_result = self.update_model_with_direction(d_t, initial_parameters)
        self.initial_parameters = ndarrays_to_parameters(adafed_result)
        parameters_aggregated = ndarrays_to_parameters(adafed_result)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Store metrics as dictionary
        my_results = metrics_aggregated
        self.results_to_save[server_round] = my_results

        # Save metrics as json
        with open(
            f"{self.results_dir}/adafed-{self.cli_strategy}-results-seed-{self.seed}.json",
            "w",
        ) as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Save new Global Model as a PyTorch checkpoint
        self._update_best_acc(
            current_round=server_round,
            accuracy=metrics_aggregated.get("avg-val-accuracy", 0.0),
            arrays=ArrayRecord.from_numpy_ndarrays(adafed_result),
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
            f"{self.results_dir}/adafed-{self.cli_strategy}-results-seed-{self.seed}.json",
            "w",
        ) as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Return the expected outputs for `evaluate`
        return loss_aggregated, metrics_aggregated

    @staticmethod
    # Modified Gram-Schmidt from https://arxiv.org/abs/2401.04993, step 1
    def modified_gram_schmidt(
        gradients: np.ndarray, losses: np.ndarray, gamma: float = 1.0
    ) -> np.ndarray:
        """Modified Gram-Schmidt with numerical stability checks."""
        K, D = gradients.shape
        ortho_grads = np.zeros_like(gradients)
        eps = 1e-10  # Add small epsilon for numerical stability

        ortho_grads[0] = gradients[0] / (np.abs(losses[0]) ** gamma + eps)

        for k in range(1, K):
            gk = gradients[k]
            fk_gamma = np.abs(losses[k]) ** gamma

            proj_sum = np.zeros_like(gk)
            for i in range(k):
                gi_tilde = ortho_grads[i]
                denom = np.dot(gi_tilde, gi_tilde)
                if denom > eps:
                    proj_sum += (np.dot(gk, gi_tilde) / denom) * gi_tilde

            numerator = gk - proj_sum

            # Compute denominator with numerical stability
            denom_correction = sum(
                np.dot(gk, ortho_grads[i])
                / (np.dot(ortho_grads[i], ortho_grads[i]) + eps)
                for i in range(k)
            )
            denominator = fk_gamma - denom_correction

            # Add clipping to prevent explosion
            if np.abs(denominator) < eps:
                # If denominator is near zero, skip normalization (keep unnormalized gradient)
                ortho_grads[k] = numerator / (eps * 10)
                logger.log(
                    WARNING, f"Near-zero denominator at k={k}, using small value"
                )
            else:
                ortho_grads[k] = numerator / denominator

        return ortho_grads

    @staticmethod
    # Convex hull minimum norm from https://arxiv.org/abs/2401.04993, step 2
    def compute_convex_combination(ortho_grads: np.ndarray):
        norm_squared = np.linalg.norm(ortho_grads, axis=1) ** 2
        alpha = 2 / np.sum(1.0 / norm_squared)  # Eq. 13
        lambdas = alpha / (2 * norm_squared)  # Eq. 12 & 14
        v_t = np.sum(lambdas[:, np.newaxis] * ortho_grads, axis=0)
        return v_t

    def update_model_with_direction(
        self, d_t: np.ndarray, base_weights: List[np.ndarray]
    ):
        # d_t is flat, base_weights is list of arrays with original shapes
        new_weights = []
        offset = 0
        for w in base_weights:
            shape, size = w.shape, np.prod(w.shape)
            delta = d_t[offset : offset + size].reshape(shape)
            new_w = w - self.lr * delta
            new_weights.append(new_w)
            offset += size
        return new_weights

    def compute_adjusted_dt(self, d_t: np.ndarray) -> np.ndarray:
        if self.use_yogi:
            # FedYogi (adaptive learning rate)
            # Initialize moment estimates if not already done
            if self.m_t is None:
                self.m_t = np.zeros_like(d_t)
            if self.v_t is None:
                self.v_t = np.zeros_like(d_t)
                logger.log(INFO, "➡️ Initialized Yogi moments")

            # FedYogi update rule based on "Adaptive Federated Optimization" from https://arxiv.org/pdf/2003.00295v5
            self.t += 1
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * d_t
            self.v_t = self.v_t - (1 - self.beta2) * (d_t**2) * np.sign(
                self.v_t - d_t**2
            )
            # self.m_that = self.m_t / (1 - self.beta1 ** self.t)  # Bias correction
            # self.v_that = self.v_t / (1 - self.beta2 ** self.t)  # Bias correction
            self.m_that = self.m_t
            self.v_that = self.v_t
            adjusted_dt = self.m_that / (np.sqrt(self.v_that) + self.tau)

        elif self.use_adam:
            # Adam (adaptive learning rate)
            # Initialize moment estimates if not already done
            if self.m_t is None:
                self.m_t = np.zeros_like(d_t)
            if self.v_t is None:
                self.v_t = np.zeros_like(d_t)
                logger.log(INFO, "➡️ Initialized Adam moments")

            # Adam update rule based on "Adaptive Federated Optimization" from https://arxiv.org/pdf/2003.00295v5
            self.t += 1
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * d_t
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (d_t**2)
            # self.m_that = self.m_t / (1 - self.beta1 ** self.t)  # Bias correction
            # self.v_that = self.v_t / (1 - self.beta2 ** self.t)  # Bias correction
            # self.m_that = self.m_t
            # self.v_that = self.v_t
            adjusted_dt = self.m_that / (np.sqrt(self.v_that) + self.tau)
        else:
            adjusted_dt = d_t

        return adjusted_dt
