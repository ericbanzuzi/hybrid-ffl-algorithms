
from logging import WARNING
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
from flwr.server.strategy import FedProx
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class AdaFedStrategy(FedProx):
    """Custom AdaFed strategy computing pseudo-gradients.

    Implementation based on "AdaFed: Fair Federated Learning via Adaptive Common Descent Direction" 
    from https://arxiv.org/abs/2401.04993.
    """

    def __init__(self, 
                 gamma: float = 1.0, 
                 eta: float = 1, 
                 use_yogi: bool = False,
                 use_adam: bool = False, 
                 beta1: float = 0.9, 
                 beta2: float = 0.99,
                 m_t: Optional[np.ndarray] = None,
                 v_t: Optional[np.ndarray] = None,
                 tau: float = 1e-10,  # Small constant for stability
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.global_weights: Optional[List[np.ndarray]] = None
        self.lr = eta
        self.use_yogi = use_yogi
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_t = m_t  # type: ignore
        self.v_t = v_t  # type: ignore
        self.tau = tau

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"AdaFed(accept_failures={self.accept_failures})"
        return rep
    
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
            # Compute pseudo-gradient as single d-dimensional vectors
            pseudo_gradient: NDArrays = np.concatenate([
                x - y
                for x, y in zip(
                    initial_parameters, parameters_to_ndarrays(fit_res.parameters)
                )
            ], axis=None)

            gradients_list.append(pseudo_gradient)
            losses.append(fit_res.metrics["train_loss"])

        gradients_matrix = np.stack(gradients_list)

        # Orthogonalize gradients
        orthograds = self.modified_gram_schmidt(gradients_matrix, losses, self.gamma)

        # Minimum norm element
        d_t = self.compute_convex_combination(orthograds)

        for i in range(min(3, len(orthograds))):
            for j in range(i+1, min(3, len(orthograds))):
                cosine = np.dot(orthograds[i], orthograds[j]) / (np.linalg.norm(orthograds[i]) * np.linalg.norm(orthograds[j]))
                print(f"Cosine between orthograd {i} and {j}: {cosine}")
        
        # print(f"Gradient shape: {gradients_matrix.shape}")
        # print(f"d_t norm: {np.linalg.norm(d_t)}")
        # print(f"Initial param norm: {np.linalg.norm(np.concatenate([p.flatten() for p in initial_parameters]))}")

        if self.use_yogi:
            # FedYogi (adaptive learning rate)
            # Initialize moment estimates if not already done
            if self.m_t is None:
                self.m_t = np.zeros_like(d_t)
            if self.v_t is None:
                self.v_t = np.zeros_like(d_t)
                self.t = 0
            
            # FedYogi update rule based on "Adaptive Federated Optimization" from https://arxiv.org/pdf/2003.00295v5
            self.t += 1
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * d_t
            self.v_t = self.v_t - (1 - self.beta2) * (d_t ** 2) * np.sign(self.v_t - d_t ** 2)
            self.m_that = self.m_t / (1 - self.beta1 ** self.t)  # Bias correction
            self.v_that = self.v_t / (1 - self.beta2 ** self.t)  # Bias correction
            # self.m_that = self.m_t
            # self.v_that = self.v_t
            adjusted_dt = self.m_that / (np.sqrt(self.v_that) + self.tau)
            
            adafed_result = self.update_model_with_direction(adjusted_dt, initial_parameters)
        elif self.use_adam:
            # Adam (adaptive learning rate)
            # Initialize moment estimates if not already done
            if self.m_t is None:
                self.m_t = np.zeros_like(d_t)
            if self.v_t is None:
                self.v_t = np.zeros_like(d_t)
                self.t = 0
            
            # Adam update rule based on "Adaptive Federated Optimization" from https://arxiv.org/pdf/2003.00295v5
            self.t += 1
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * d_t
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (d_t ** 2)
            self.m_that = self.m_t / (1 - self.beta1 ** self.t)  # Bias correction
            self.v_that = self.v_t / (1 - self.beta2 ** self.t)  # Bias correction
            # self.m_that = self.m_t
            # self.v_that = self.v_t
            adjusted_dt = self.m_that / (np.sqrt(self.v_that) + self.tau)
            
            adafed_result = self.update_model_with_direction(adjusted_dt, initial_parameters)

        else:
            # Standard SGD update
            adafed_result = self.update_model_with_direction(d_t, initial_parameters)

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

    @staticmethod
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
    

    @staticmethod
    # Convex hull minimum norm from https://arxiv.org/abs/2401.04993, step 2
    def compute_convex_combination(ortho_grads: np.ndarray):
        norm_squared = np.linalg.norm(ortho_grads, axis=1) ** 2
        alpha = 2 / np.sum(1.0 / norm_squared)  # Eq. 13
        lambdas = alpha / (2 * norm_squared)  # Eq. 12 & 14
        v_t = np.sum(lambdas[:, np.newaxis] * ortho_grads, axis=0)
        return v_t
    
    def update_model_with_direction(self, d_t: np.ndarray, base_weights: List[np.ndarray]):
        # d_t is flat, base_weights is list of arrays with original shapes
        new_weights = []
        offset = 0
        for w in base_weights:
            shape, size = w.shape, np.prod(w.shape)
            delta = d_t[offset:offset+size].reshape(shape)
            if self.use_yogi: 
                new_w = w + self.lr * delta  # FedYogi uses a "+" update rule
            elif self.use_adam:
                new_w = w + self.lr * delta # Adam also uses a "+" update rule
            else:
                new_w = w - self.lr * delta
            new_weights.append(new_w)
            offset += size
        return new_weights
