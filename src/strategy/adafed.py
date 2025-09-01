
from logging import WARNING
import flwr as fl
import numpy as np
from typing import List, Optional, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


# Modified Gram-Schmidt from https://arxiv.org/abs/2401.04993, step 1
def modified_gram_schmidt(gradients: np.ndarray, losses: np.ndarray, gamma: float = 1.0):
    K, D = gradients.shape
    ortho_grads = np.zeros_like(gradients)

    # Eq. 5
    ortho_grads[0] = gradients[0] / (np.abs(losses[0]) ** gamma)

    # Eq. 6
    for k in range(1, K):
        gk = gradients[k]
        fk_gamma = np.abs(losses[k]) ** gamma

        proj_sum = np.zeros_like(gk)
        for i in range(k):
            gi_tilde = ortho_grads[i]
            proj_sum += (np.dot(gk, gi_tilde) / np.dot(gi_tilde, gi_tilde)) * gi_tilde

        numerator = gk - proj_sum
        denominator = fk_gamma - sum(
            np.dot(gk, ortho_grads[i]) / np.dot(ortho_grads[i], ortho_grads[i]) for i in range(k)
        )
        ortho_grads[k] = numerator / denominator

    return ortho_grads


# Convex hull minimum norm from https://arxiv.org/abs/2401.04993, step 2
def compute_convex_combination(ortho_grads: np.ndarray):
    norm_squared = np.linalg.norm(ortho_grads, axis=1) ** 2
    alpha = 2 / np.sum(1.0 / norm_squared)  # Eq. 13
    lambdas = alpha / (2 * norm_squared)  # Eq. 12 & 14
    v_t = np.sum(lambdas[:, np.newaxis] * ortho_grads, axis=0)
    return v_t


# Custom AdaFed Strategy
class AdaFedStrategy2(fl.server.strategy.FedProx):
    def __init__(self, eta: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = eta

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        gradients_list = []
        losses = []

        for _, fit_res in results:
            gradients = parameters_to_ndarrays(fit_res.parameters)
            loss = fit_res.metrics["loss"] if "loss" in fit_res.metrics else 1.0
            flat_grad = np.concatenate([g.flatten() for g in gradients])
            gradients_list.append(flat_grad)
            losses.append(loss)

        gradients_matrix = np.stack(gradients_list)
        losses_array = np.array(losses)

        # Orthogonalize
        ortho_grads = modified_gram_schmidt(gradients_matrix, losses_array)

        # Minimum norm element
        v_t = compute_convex_combination(ortho_grads)

        # Update global parameters
        flat_params = np.concatenate([p.flatten() for p in parameters_to_ndarrays(self.current_parameters)])
        updated_params = flat_params - self.eta * v_t

        shapes = [p.shape for p in parameters_to_ndarrays(self.current_parameters)]
        sizes = [np.prod(shape) for shape in shapes]
        split_indices = np.cumsum(sizes)[:-1]
        updated_param_list = np.split(updated_params, split_indices)
        updated_param_list = [param.reshape(shape) for param, shape in zip(updated_param_list, shapes)]

        self.current_parameters = ndarrays_to_parameters(updated_param_list)

        return self.current_parameters, {}
    

class AdaFedStrategy(fl.server.strategy.FedProx):
    """Custom AdaFed strategy computing pseudo-gradients."""

    def __init__(self, gamma: float = 1.0, eta: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.global_weights: Optional[List[np.ndarray]] = None
        self.lr = eta
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters
    
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
            pseudo_gradient: NDArrays = np.concatenate([
                x - y
                for x, y in zip(
                    initial_parameters, parameters_to_ndarrays(fit_res.parameters)
                )
            ], axis=None)
            gradients_list.append(pseudo_gradient)
            losses.append(fit_res.metrics["train_loss"])

        gradients_matrix = np.stack(gradients_list)

        # Per-client: get (parameters, loss)
        # pseudo_grads_losses = [
        #     (
        #         [
        #             client_layer - init_layer
        #             for client_layer, init_layer in zip(parameters_to_ndarrays(fit_res.parameters), initial_ndarrays)
        #         ],
        #         fit_res.metrics['loss'],
        #     )
        #     for _, fit_res in results
        # ]

        orthograds = modified_gram_schmidt(gradients_matrix, losses, self.gamma)
        d_t = compute_convex_combination(orthograds)

        for i in range(min(3, len(orthograds))):
            for j in range(i+1, min(3, len(orthograds))):
                cosine = np.dot(orthograds[i], orthograds[j]) / (np.linalg.norm(orthograds[i]) * np.linalg.norm(orthograds[j]))
                print(f"Cosine between orthograd {i} and {j}: {cosine}")
        
        print(f"Gradient shape: {gradients_matrix.shape}")
        print(f"d_t norm: {np.linalg.norm(d_t)}")
        print(f"Initial param norm: {np.linalg.norm(np.concatenate([p.flatten() for p in initial_parameters]))}")

        adafed_result = self.update_model_with_direction(d_t, initial_parameters)


        # fedavg_result = aggregate(weights_results)

        # following convention described in
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        # if self.server_opt:
        #     # You need to initialize the model
        #     assert (
        #         self.initial_parameters is not None
        #     ), "When using server-side optimization, model needs to be initialized."
        #     initial_weights = parameters_to_ndarrays(self.initial_parameters)

        #     # remember that updates are the opposite of gradients
        #     pseudo_gradient: NDArrays = [
        #         x - y
        #         for x, y in zip(
        #             parameters_to_ndarrays(self.initial_parameters), fedavg_result
        #         )
        #     ]
        #     if self.server_momentum > 0.0:
        #         if server_round > 1:
        #             assert (
        #                 self.momentum_vector
        #             ), "Momentum should have been created on round 1."
        #             self.momentum_vector = [
        #                 self.server_momentum * x + y
        #                 for x, y in zip(self.momentum_vector, pseudo_gradient)
        #             ]
        #         else:
        #             self.momentum_vector = pseudo_gradient

        #         # No nesterov for now
        #         pseudo_gradient = self.momentum_vector

        #     # SGD
        #     fedavg_result = [
        #         x - self.server_learning_rate * y
        #         for x, y in zip(initial_weights, pseudo_gradient)
        #     ]
        # Update current weights
        self.initial_parameters = ndarrays_to_parameters(adafed_result)

        parameters_aggregated = ndarrays_to_parameters(adafed_result)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_fit2(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # Extract client weights
        client_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        # Extract number of examples per client
        num_examples = np.array([fit_res.num_examples for _, fit_res in results])

        # Compute global weights (previous round)
        if self.global_weights is None:
            # First round: initialize from any client
            self.global_weights = client_weights[0]

        # Compute pseudo-gradients: delta = client_weights - global_weights
        pseudo_grads = [
            [cw_i - gw_i for cw_i, gw_i in zip(client_weight, self.global_weights)]
            for client_weight in client_weights
        ]

        log(np.array(pseudo_grads).shape)

        log('after pseudo')
        # Stack pseudo-grads as 2D arrays for orthogonalization
        grads_flat = np.array([
            np.concatenate([g.flatten() for g in pseudo_grad])
            for pseudo_grad in pseudo_grads
        ])  # Shape: (num_clients, total_params)

        # Dummy losses: simulate (normally you get real ones)
        dummy_losses = np.random.uniform(0.5, 1.5, size=len(results))

        # Modified Gram-Schmidt orthogonalization
        ortho_grads = self.modified_gram_schmidt(grads_flat, dummy_losses, gamma=self.gamma)

        # Convex combination to get common direction
        d_t = self.compute_convex_combination(ortho_grads)

        log(d_t.shape)

        new_weights = self.global_weights - self.lr * d_t
        # # Reshape v_t to model format
        # new_weights = self.update_model_with_direction(v_t, self.global_weights)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        # # Save for next round
        # self.global_weights = new_weights

        # Return new global model
        return ndarrays_to_parameters(new_weights), metrics_aggregated

    @staticmethod
    def modified_gram_schmidt(gradients: np.ndarray, losses: list, gamma: float = 1.0):
        K, D = gradients.shape
        ortho_grads = np.zeros_like(gradients)
        fk_gamma = np.abs(losses) ** gamma

        for k in range(K):
            gk = gradients[k]
            proj = np.zeros_like(gk)
            scalar_sum = 0.0
            for i in range(k):
                gi = ortho_grads[i]
                dot = np.dot(gk, gi)
                norm_sq = np.dot(gi, gi)
                proj += (dot / norm_sq) * gi
                scalar_sum += dot / norm_sq

            numerator = gk - proj
            denominator = fk_gamma[k] - scalar_sum
            ortho_grads[k] = numerator / denominator

        return ortho_grads

    @staticmethod
    def compute_convex_combination(ortho_grads: np.ndarray):
        norm_squared = np.linalg.norm(ortho_grads, axis=1) ** 2
        inv_norms = 1.0 / norm_squared
        lambdas = inv_norms / np.sum(inv_norms)
        d_t = np.sum(lambdas[:, np.newaxis] * ortho_grads, axis=0)
        return d_t

    def update_model_with_direction(self, d_t: np.ndarray, base_weights: List[np.ndarray]):
        # d_t is flat, base_weights is list of arrays
        new_weights = []
        offset = 0
        for w in base_weights:
            shape = w.shape
            size = np.prod(shape)
            delta = d_t[offset:offset+size].reshape(shape)
            new_w = w - self.lr * delta
            new_weights.append(new_w)
            offset += size
        return new_weights
