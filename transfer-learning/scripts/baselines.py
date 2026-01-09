"""
Baselines para comparación con FSBO.

Implementa los métodos de referencia estándar en HPO:
1. Random Search - Muestreo aleatorio uniforme
2. GP-LHS - GP con Latin Hypercube Sampling
3. GP-RS - GP vanilla con Random Sampling

Referencias:
- Bergstra & Bengio (2012) - Random Search for HPO
- Snoek et al. (2012) - Practical Bayesian Optimization

Autor: Proyecto académico MetaLearning
"""

import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from scipy.stats import norm
from scipy.stats import qmc  # Latin Hypercube Sampling
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Estructuras comunes
# =============================================================================

@dataclass
class OptimizationTrace:
    """Traza de una optimización."""
    method: str
    observed_x: List[np.ndarray]
    observed_y: List[float]
    best_y_history: List[float]
    
    @property
    def best_y(self) -> float:
        return max(self.observed_y) if self.observed_y else 0.0
    
    @property
    def n_evaluations(self) -> int:
        return len(self.observed_y)


# =============================================================================
# Baseline 1: Random Search
# =============================================================================

class RandomSearch:
    """
    Random Search baseline.
    
    Simplemente muestrea configuraciones uniformemente al azar.
    Sorprendentemente efectivo y difícil de superar en espacios de baja dimensión.
    
    Reference:
        Bergstra & Bengio (2012) - "Random Search for Hyper-Parameter Optimization"
    """
    
    def __init__(self, input_dim: int, seed: Optional[int] = None):
        self.input_dim = input_dim
        self.rng = np.random.RandomState(seed)
        self.observed_x: List[np.ndarray] = []
        self.observed_y: List[float] = []
        self.best_y_history: List[float] = []
    
    def suggest(self) -> np.ndarray:
        """Sugiere una configuración aleatoria uniformemente."""
        return self.rng.rand(self.input_dim).astype(np.float32)
    
    def observe(self, x: np.ndarray, y: float):
        """Registra una observación."""
        self.observed_x.append(x)
        self.observed_y.append(y)
        
        best_so_far = max(self.observed_y)
        self.best_y_history.append(best_so_far)
    
    def get_trace(self) -> OptimizationTrace:
        return OptimizationTrace(
            method="Random Search",
            observed_x=self.observed_x.copy(),
            observed_y=self.observed_y.copy(),
            best_y_history=self.best_y_history.copy()
        )
    
    def reset(self):
        self.observed_x = []
        self.observed_y = []
        self.best_y_history = []


# =============================================================================
# GP Vanilla (base para GP-LHS y GP-RS)
# =============================================================================

class VanillaGP(ExactGP):
    """
    Gaussian Process vanilla con kernel RBF.
    
    A diferencia de FSBO, no tiene deep kernel ni pre-entrenamiento.
    Cada tarea empieza desde cero.
    """
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[-1]))
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class GPOptimizer:
    """
    Optimizador basado en GP vanilla.
    
    Base para GP-LHS y GP-RS.
    """
    
    def __init__(
        self,
        input_dim: int,
        seed: Optional[int] = None,
        n_restarts: int = 5,
        device: str = 'cpu'
    ):
        self.input_dim = input_dim
        self.rng = np.random.RandomState(seed)
        self.n_restarts = n_restarts
        self.device = device
        
        self.observed_x: List[np.ndarray] = []
        self.observed_y: List[float] = []
        self.best_y_history: List[float] = []
        
        self.model = None
        self.likelihood = None
    
    def _fit_gp(self):
        """Ajusta el GP a las observaciones actuales."""
        if len(self.observed_x) < 2:
            return
        
        X = torch.tensor(np.array(self.observed_x), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(self.observed_y), dtype=torch.float32).to(self.device)
        
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = VanillaGP(X, y, self.likelihood).to(self.device)
        
        # Entrenar GP
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for _ in range(50):  # Pocas iteraciones
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
    
    def _expected_improvement(
        self,
        X_candidates: torch.Tensor,
        y_best: float,
        xi: float = 0.01
    ) -> np.ndarray:
        """Calcula Expected Improvement."""
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_candidates))
            mu = pred.mean.cpu().numpy()
            sigma = pred.stddev.cpu().numpy()
        
        sigma = np.maximum(sigma, 1e-8)
        improvement = mu - y_best - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-8] = 0.0
        
        return ei
    
    def suggest(self, n_candidates: int = 1000) -> np.ndarray:
        """Sugiere siguiente configuración usando EI."""
        if len(self.observed_x) < 2:
            # Sin suficientes datos, muestrear aleatorio
            return self.rng.rand(self.input_dim).astype(np.float32)
        
        # Ajustar GP
        self._fit_gp()
        
        # Generar candidatos
        X_candidates = self.rng.rand(n_candidates, self.input_dim).astype(np.float32)
        X_candidates_tensor = torch.tensor(X_candidates).to(self.device)
        
        # Calcular EI
        y_best = max(self.observed_y)
        ei = self._expected_improvement(X_candidates_tensor, y_best)
        
        # Seleccionar mejor
        best_idx = np.argmax(ei)
        return X_candidates[best_idx]
    
    def observe(self, x: np.ndarray, y: float):
        """Registra una observación."""
        self.observed_x.append(x)
        self.observed_y.append(y)
        
        best_so_far = max(self.observed_y)
        self.best_y_history.append(best_so_far)
    
    def get_trace(self) -> OptimizationTrace:
        return OptimizationTrace(
            method="GP",
            observed_x=self.observed_x.copy(),
            observed_y=self.observed_y.copy(),
            best_y_history=self.best_y_history.copy()
        )
    
    def reset(self):
        self.observed_x = []
        self.observed_y = []
        self.best_y_history = []
        self.model = None
        self.likelihood = None


# =============================================================================
# Baseline 2: GP con Latin Hypercube Sampling
# =============================================================================

class GP_LHS(GPOptimizer):
    """
    GP con inicialización Latin Hypercube Sampling.
    
    LHS asegura una cobertura más uniforme del espacio que muestreo aleatorio.
    Es el estándar en muchos frameworks de BO.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_init: int = 5,
        seed: Optional[int] = None,
        device: str = 'cpu'
    ):
        super().__init__(input_dim, seed, device=device)
        self.n_init = n_init
        self._initial_points = None
        self._init_idx = 0
    
    def _generate_lhs(self, n_samples: int) -> np.ndarray:
        """Genera muestras usando Latin Hypercube Sampling."""
        sampler = qmc.LatinHypercube(d=self.input_dim, seed=self.rng.randint(0, 2**31))
        samples = sampler.random(n=n_samples)
        return samples.astype(np.float32)
    
    def suggest(self, n_candidates: int = 1000) -> np.ndarray:
        """Sugiere siguiente configuración."""
        # Primero usar puntos LHS
        if self._initial_points is None:
            self._initial_points = self._generate_lhs(self.n_init)
            self._init_idx = 0
        
        if self._init_idx < self.n_init:
            point = self._initial_points[self._init_idx]
            self._init_idx += 1
            return point
        
        # Luego usar GP con EI
        return super().suggest(n_candidates)
    
    def get_trace(self) -> OptimizationTrace:
        trace = super().get_trace()
        trace.method = "GP-LHS"
        return trace
    
    def reset(self):
        super().reset()
        self._initial_points = None
        self._init_idx = 0


# =============================================================================
# Baseline 3: GP con Random Sampling
# =============================================================================

class GP_RS(GPOptimizer):
    """
    GP con inicialización Random Sampling.
    
    Similar a GP-LHS pero con inicialización puramente aleatoria.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_init: int = 5,
        seed: Optional[int] = None,
        device: str = 'cpu'
    ):
        super().__init__(input_dim, seed, device=device)
        self.n_init = n_init
        self._n_random = 0
    
    def suggest(self, n_candidates: int = 1000) -> np.ndarray:
        """Sugiere siguiente configuración."""
        # Primero puntos aleatorios
        if self._n_random < self.n_init:
            self._n_random += 1
            return self.rng.rand(self.input_dim).astype(np.float32)
        
        # Luego usar GP con EI
        return super().suggest(n_candidates)
    
    def get_trace(self) -> OptimizationTrace:
        trace = super().get_trace()
        trace.method = "GP-RS"
        return trace
    
    def reset(self):
        super().reset()
        self._n_random = 0


# =============================================================================
# Factory para crear optimizadores
# =============================================================================

def create_optimizer(
    method: str,
    input_dim: int,
    n_init: int = 5,
    seed: Optional[int] = None,
    device: str = 'cpu'
):
    """
    Factory para crear optimizadores.
    
    Args:
        method: 'random', 'gp-lhs', 'gp-rs'
        input_dim: Dimensión del espacio de búsqueda
        n_init: Configuraciones iniciales (para GP)
        seed: Semilla aleatoria
        device: Dispositivo para GP
        
    Returns:
        Optimizador configurado
    """
    method = method.lower()
    
    if method == 'random':
        return RandomSearch(input_dim, seed)
    elif method == 'gp-lhs':
        return GP_LHS(input_dim, n_init, seed, device)
    elif method == 'gp-rs':
        return GP_RS(input_dim, n_init, seed, device)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Función para ejecutar optimización con cualquier baseline
# =============================================================================

def run_baseline(
    method: str,
    evaluate_fn: Callable[[np.ndarray], float],
    input_dim: int,
    n_trials: int = 50,
    n_init: int = 5,
    seed: Optional[int] = None,
    device: str = 'cpu',
    verbose: bool = False
) -> OptimizationTrace:
    """
    Ejecuta una optimización con un baseline.
    
    Args:
        method: 'random', 'gp-lhs', 'gp-rs'
        evaluate_fn: Función fn(x) -> score
        input_dim: Dimensión del espacio
        n_trials: Número total de evaluaciones
        n_init: Configuraciones iniciales
        seed: Semilla aleatoria
        device: Dispositivo
        verbose: Mostrar progreso
        
    Returns:
        OptimizationTrace con resultados
    """
    optimizer = create_optimizer(method, input_dim, n_init, seed, device)
    
    for i in range(n_trials):
        x = optimizer.suggest()
        y = evaluate_fn(x)
        optimizer.observe(x, y)
        
        if verbose and (i + 1) % 10 == 0:
            best = max(optimizer.observed_y)
            print(f"  [{method}] Trial {i+1}/{n_trials}: best={best:.4f}")
    
    return optimizer.get_trace()


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing baselines module...")
    
    # Función de prueba: Rosenbrock 2D invertida (maximizar)
    def test_function(x):
        # Rosenbrock tiene mínimo en (1,1), invertimos para maximizar
        a, b = 1, 100
        x1, x2 = x[0] * 4 - 2, x[1] * 4 - 2  # Escalar a [-2, 2]
        val = (a - x1)**2 + b * (x2 - x1**2)**2
        return 1 / (1 + val)  # Invertir y normalizar
    
    input_dim = 2
    n_trials = 30
    
    results = {}
    
    for method in ['random', 'gp-lhs', 'gp-rs']:
        print(f"\nRunning {method}...")
        trace = run_baseline(
            method=method,
            evaluate_fn=test_function,
            input_dim=input_dim,
            n_trials=n_trials,
            n_init=5,
            seed=42,
            verbose=True
        )
        results[method] = trace
        print(f"  Final best: {trace.best_y:.4f}")
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    for method, trace in results.items():
        print(f"{method:<15} Best: {trace.best_y:.4f} ({trace.n_evaluations} evals)")
    
    print("\n✅ Baselines module tests passed!")

