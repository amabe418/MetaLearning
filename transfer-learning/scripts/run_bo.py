"""
Script para ejecutar Bayesian Optimization con modelo FSBO pre-entrenado.

Implementa el Algoritmo 2 del paper:
    "Few-Shot Bayesian Optimization with Deep Kernel Surrogates"
    Wistuba & Grabocka (ICLR 2021)

Uso:
    python scripts/run_bo.py --checkpoint fsbo_adaboost_20260107_151935.pt
    python scripts/run_bo.py --checkpoint fsbo_adaboost_20260107_151935.pt --n_trials 30
    python scripts/run_bo.py --algorithm adaboost --n_trials 50

Autor: Proyecto académico MetaLearning
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from scipy.stats import norm
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Modelos (mismos que en train_fsbo.py)
# =============================================================================

class DeepKernelNetwork(nn.Module):
    """
    Red neuronal que transforma hiperparámetros a espacio latente.
    
    Paper Sección 3.1:
        "We propose to use a neural network φ to project the hyperparameters 
        into a latent space where a standard kernel can be applied."
    
    Ecuación (3): k_DK(x, x') = k(φ(x), φ(x'))
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim
        
    def forward(self, x):
        return self.network(x)


class DeepKernelGP(ExactGP):
    """
    Gaussian Process con Deep Kernel.
    
    Paper Sección 3.1:
        "The surrogate is a Gaussian process with a deep kernel"
        f(x) ~ GP(μ, k_DK)
    """
    
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = ConstantMean()
        latent_dim = feature_extractor.output_dim
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean = self.mean_module(projected_x)
        covar = self.covar_module(projected_x)
        return MultivariateNormal(mean, covar)


# =============================================================================
# Expected Improvement - Acquisition Function
# =============================================================================

def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Calcula Expected Improvement para cada candidato.
    
    Paper Sección 2 (Background on Bayesian Optimization):
        "An acquisition function is used that balances the trade-off between 
        exploration and exploitation."
    
    EI(x) = (μ(x) - y_best - ξ) * Φ(Z) + σ(x) * φ(Z)
    donde Z = (μ(x) - y_best - ξ) / σ(x)
    
    Args:
        mu: Predicciones de media del GP (n_candidates,)
        sigma: Predicciones de desviación estándar (n_candidates,)
        y_best: Mejor valor observado hasta ahora
        xi: Factor de exploración (default 0.01)
        
    Returns:
        EI values (n_candidates,)
    """
    # Evitar división por cero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calcular Z (improvement normalizado)
    improvement = mu - y_best - xi
    Z = improvement / sigma
    
    # EI = improvement * Φ(Z) + σ * φ(Z)
    # Φ = CDF normal, φ = PDF normal
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # EI es 0 donde σ es muy pequeño
    ei[sigma < 1e-8] = 0.0
    
    return ei


# =============================================================================
# Warm Start - Selección de configuraciones iniciales
# =============================================================================

def warm_start_random(X_pool: np.ndarray, n_init: int = 5) -> np.ndarray:
    """
    Warm start simple: selección aleatoria.
    
    Paper Sección 3.4 menciona un algoritmo evolutivo, pero para simplicidad
    usamos selección aleatoria diversa (Latin Hypercube-like).
    
    Args:
        X_pool: Pool de configuraciones candidatas (n_candidates, hp_dim)
        n_init: Número de configuraciones iniciales
        
    Returns:
        Índices de las configuraciones seleccionadas
    """
    n_candidates = len(X_pool)
    
    if n_init >= n_candidates:
        return np.arange(n_candidates)
    
    # Selección diversa: maximizar distancia entre puntos
    selected = [np.random.randint(n_candidates)]
    
    for _ in range(n_init - 1):
        # Calcular distancia mínima a puntos ya seleccionados
        min_distances = np.full(n_candidates, np.inf)
        
        for idx in selected:
            distances = np.linalg.norm(X_pool - X_pool[idx], axis=1)
            min_distances = np.minimum(min_distances, distances)
        
        # Seleccionar el punto más lejano
        min_distances[selected] = -np.inf  # Excluir ya seleccionados
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)
    
    return np.array(selected)


def warm_start_model_based(
    model: DeepKernelGP,
    likelihood: GaussianLikelihood,
    X_pool: torch.Tensor,
    n_init: int = 5,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Warm start basado en el modelo pre-entrenado.
    
    Paper Sección 3.4:
        "We propose to use the pre-trained surrogate to select the initial 
        configurations... selecting configurations that are predicted to 
        perform well."
    
    Selecciona configuraciones con alta media predicha (explotación inicial).
    
    Args:
        model: Modelo FSBO pre-entrenado
        likelihood: Likelihood del GP
        X_pool: Pool de configuraciones candidatas
        n_init: Número de configuraciones iniciales
        device: Dispositivo (cpu/cuda)
        
    Returns:
        Índices de las configuraciones seleccionadas
    """
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Predecir para todo el pool
        X_pool = X_pool.to(device)
        pred = likelihood(model(X_pool))
        mu = pred.mean.cpu().numpy()
    
    # Seleccionar los n_init con mayor media predicha
    # Pero añadimos algo de diversidad
    top_indices = np.argsort(mu)[-n_init * 3:]  # Top 3x candidatos
    
    # De esos, seleccionar diversamente
    selected = warm_start_random(X_pool.cpu().numpy()[top_indices], n_init)
    
    return top_indices[selected]


# =============================================================================
# Fine-tuning
# =============================================================================

def finetune_model(
    model: DeepKernelGP,
    likelihood: GaussianLikelihood,
    X_observed: torch.Tensor,
    y_observed: torch.Tensor,
    n_epochs: int = 20,
    lr: float = 1e-4,
    device: str = 'cpu'
):
    """
    Fine-tuning del modelo en la tarea objetivo.
    
    Paper Sección 3.3:
        "After each new observation, we fine-tune the pre-trained surrogate 
        on the observations of the target task."
        
        "We use a small learning rate (10^-4) and few epochs to avoid 
        overfitting to the small number of observations."
    
    Args:
        model: Modelo FSBO pre-entrenado
        likelihood: Likelihood del GP
        X_observed: Configuraciones observadas (n_obs, hp_dim)
        y_observed: Scores observados (n_obs,)
        n_epochs: Número de epochs de fine-tuning
        lr: Learning rate (paper: 10^-4)
        device: Dispositivo
    """
    model.train()
    likelihood.train()
    
    # Actualizar datos del GP
    model.set_train_data(X_observed.to(device), y_observed.to(device), strict=False)
    
    # Optimizador con lr bajo
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.covar_module.parameters(), 'lr': lr},
        {'params': model.mean_module.parameters(), 'lr': lr},
        {'params': likelihood.parameters(), 'lr': lr},
    ])
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_observed.to(device))
        loss = -mll(output, y_observed.to(device))
        loss.backward()
        optimizer.step()


# =============================================================================
# Loop de Bayesian Optimization
# =============================================================================

def run_bo_loop(
    model: DeepKernelGP,
    likelihood: GaussianLikelihood,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    n_trials: int = 50,
    n_init: int = 5,
    finetune_frequency: int = 5,
    finetune_epochs: int = 20,
    n_candidates: int = 1000,
    xi: float = 0.01,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    Ejecuta el loop de Bayesian Optimization.
    
    Paper Algoritmo 2 (BO with FSBO):
        1. Initialize D₀ with warm start
        2. for t = 1 to T do
        3.     Fine-tune surrogate on Dₜ₋₁
        4.     Select xₜ = argmax EI(x)
        5.     Evaluate yₜ = f(xₜ)
        6.     Dₜ = Dₜ₋₁ ∪ {(xₜ, yₜ)}
        7. end for
        8. return best configuration
    
    Args:
        model: Modelo FSBO pre-entrenado
        likelihood: Likelihood del GP
        X_pool: Pool de todas las configuraciones posibles
        y_pool: Scores verdaderos (para simular evaluación)
        n_trials: Número total de evaluaciones (presupuesto)
        n_init: Número de configuraciones iniciales
        finetune_frequency: Fine-tune cada N trials
        finetune_epochs: Epochs de fine-tuning
        n_candidates: Candidatos a evaluar por EI
        xi: Factor de exploración para EI
        device: Dispositivo
        verbose: Mostrar progreso
        
    Returns:
        Diccionario con resultados del BO
    """
    n_pool = len(X_pool)
    
    # Convertir a tensores
    X_pool_tensor = torch.tensor(X_pool, dtype=torch.float32)
    
    # Track de observaciones
    observed_indices = []
    observed_X = []
    observed_y = []
    
    # Histórico
    best_y_history = []
    ei_history = []
    
    # =========================================================================
    # PASO 1: Warm Start (Paper Sección 3.4)
    # =========================================================================
    if verbose:
        print(f"\n🚀 Warm Start: Seleccionando {n_init} configuraciones iniciales...")
    
    # Usar warm start basado en modelo
    init_indices = warm_start_model_based(
        model, likelihood, X_pool_tensor, n_init, device
    )
    
    for idx in init_indices:
        observed_indices.append(idx)
        observed_X.append(X_pool[idx])
        observed_y.append(y_pool[idx])
    
    best_y = max(observed_y)
    best_y_history.extend([best_y] * n_init)
    
    if verbose:
        print(f"   Mejor inicial: {best_y:.4f}")
    
    # =========================================================================
    # PASO 2: Loop de BO (Paper Algoritmo 2)
    # =========================================================================
    remaining_trials = n_trials - n_init
    
    pbar = tqdm(range(remaining_trials), desc="BO Loop", disable=not verbose)
    
    for trial in pbar:
        # Convertir observaciones a tensores
        X_obs_tensor = torch.tensor(np.array(observed_X), dtype=torch.float32)
        y_obs_tensor = torch.tensor(np.array(observed_y), dtype=torch.float32)
        
        # =================================================================
        # Fine-tuning (Paper Sección 3.3)
        # =================================================================
        if (trial + 1) % finetune_frequency == 0 or trial == 0:
            finetune_model(
                model, likelihood,
                X_obs_tensor, y_obs_tensor,
                n_epochs=finetune_epochs,
                lr=1e-4,
                device=device
            )
        
        # =================================================================
        # Predecir con el modelo (Paper Sección 3.1)
        # =================================================================
        model.eval()
        likelihood.eval()
        
        # Actualizar datos del GP para predicción
        model.set_train_data(
            X_obs_tensor.to(device), 
            y_obs_tensor.to(device), 
            strict=False
        )
        
        # Generar candidatos (excluir ya observados)
        available_indices = [i for i in range(n_pool) if i not in observed_indices]
        
        if len(available_indices) == 0:
            if verbose:
                print("   ⚠️ Pool agotado")
            break
        
        # Muestrear candidatos
        n_cand = min(n_candidates, len(available_indices))
        candidate_indices = np.random.choice(available_indices, n_cand, replace=False)
        X_candidates = torch.tensor(X_pool[candidate_indices], dtype=torch.float32)
        
        # Predecir
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(X_candidates.to(device)))
            mu = pred.mean.cpu().numpy()
            sigma = pred.stddev.cpu().numpy()
        
        # =================================================================
        # Acquisition Function: Expected Improvement (Paper Sección 2)
        # =================================================================
        ei_values = expected_improvement(mu, sigma, best_y, xi=xi)
        ei_history.append(ei_values.max())
        
        # Seleccionar mejor candidato
        best_candidate_idx = np.argmax(ei_values)
        next_idx = candidate_indices[best_candidate_idx]
        
        # =================================================================
        # Evaluar (simular con y_pool)
        # =================================================================
        next_y = y_pool[next_idx]
        
        observed_indices.append(next_idx)
        observed_X.append(X_pool[next_idx])
        observed_y.append(next_y)
        
        # Actualizar mejor
        if next_y > best_y:
            best_y = next_y
        
        best_y_history.append(best_y)
        
        # Actualizar progress bar
        pbar.set_postfix({
            'best': f'{best_y:.4f}',
            'current': f'{next_y:.4f}',
            'EI': f'{ei_values.max():.4f}'
        })
    
    # =========================================================================
    # Resultados
    # =========================================================================
    best_idx = observed_indices[np.argmax(observed_y)]
    
    return {
        'best_x': X_pool[best_idx],
        'best_y': max(observed_y),
        'best_idx': best_idx,
        'observed_indices': observed_indices,
        'observed_y': observed_y,
        'best_y_history': best_y_history,
        'n_evaluations': len(observed_y),
        'y_pool_best': y_pool.max(),
        'regret': y_pool.max() - max(observed_y)
    }


# =============================================================================
# Cargar modelo desde checkpoint
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    input_dim: int,
    device: str = 'cpu'
) -> tuple:
    """
    Carga modelo FSBO desde checkpoint.
    
    Args:
        checkpoint_path: Ruta al archivo .pt
        input_dim: Dimensión de entrada (número de hiperparámetros)
        device: Dispositivo
        
    Returns:
        (model, likelihood, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    hidden_dim = config.get('hidden_dim', 128)
    
    # Crear modelo
    train_x = torch.zeros(1, input_dim).to(device)
    train_y = torch.zeros(1).to(device)
    
    feature_extractor = DeepKernelNetwork(input_dim, hidden_dim).to(device)
    likelihood = GaussianLikelihood().to(device)
    model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor).to(device)
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state'])
    likelihood.load_state_dict(checkpoint['likelihood_state'])
    
    return model, likelihood, config


# =============================================================================
# Main
# =============================================================================

ALGORITHM_FILES = {
    'adaboost': 'adaboost_target_representation_with_scores.csv',
    'random_forest': 'random_forest_target_representation_with_scores.csv',
    'libsvm_svc': 'libsvm_svc_target_representation_with_scores.csv',
    'autosklearn': 'autosklearn_target_representation_with_scores.csv',
}


def main():
    parser = argparse.ArgumentParser(description='Ejecutar BO con modelo FSBO')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Nombre del checkpoint (en experiments/checkpoints/)')
    parser.add_argument('--algorithm', type=str, default='adaboost',
                       choices=list(ALGORITHM_FILES.keys()))
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Número total de evaluaciones')
    parser.add_argument('--n_init', type=int, default=5,
                       help='Configuraciones iniciales (warm start)')
    parser.add_argument('--target_task', type=int, default=None,
                       help='ID de tarea objetivo (si no se especifica, usa una aleatoria)')
    parser.add_argument('--finetune_freq', type=int, default=5,
                       help='Fine-tune cada N trials')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Seed para reproducibilidad
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Rutas
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'representation_with_scores'
    checkpoint_dir = base_dir / 'experiments' / 'checkpoints'
    results_dir = base_dir / 'experiments' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("🎯 FSBO Bayesian Optimization")
    print("=" * 60)
    
    # =========================================================================
    # Cargar datos
    # =========================================================================
    data_file = data_dir / ALGORITHM_FILES[args.algorithm]
    
    if not data_file.exists():
        print(f"❌ Archivo no encontrado: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    
    # Identificar columnas
    task_col = 'task_id'
    score_col = 'accuracy'
    hp_cols = [c for c in df.columns if c not in [task_col, score_col]]
    
    # Obtener tareas únicas
    task_ids = df[task_col].unique()
    
    # Seleccionar tarea objetivo
    if args.target_task is not None:
        target_task = args.target_task
    else:
        # Usar una tarea aleatoria como "nueva tarea"
        target_task = np.random.choice(task_ids)
    
    print(f"\n📊 Configuración:")
    print(f"   Algoritmo: {args.algorithm}")
    print(f"   Tarea objetivo: {target_task}")
    print(f"   Trials totales: {args.n_trials}")
    print(f"   Warm start: {args.n_init}")
    print(f"   Device: {device}")
    
    # Filtrar datos de la tarea objetivo
    task_df = df[df[task_col] == target_task]
    X_pool = task_df[hp_cols].values.astype(np.float32)
    y_pool = task_df[score_col].values.astype(np.float32)
    
    print(f"\n   Pool size: {len(X_pool)} configuraciones")
    print(f"   HP dims: {X_pool.shape[1]}")
    print(f"   y range: [{y_pool.min():.4f}, {y_pool.max():.4f}]")
    print(f"   Óptimo real: {y_pool.max():.4f}")
    
    # =========================================================================
    # Cargar modelo
    # =========================================================================
    if args.checkpoint:
        checkpoint_path = checkpoint_dir / args.checkpoint
    else:
        # Buscar checkpoint más reciente para el algoritmo
        checkpoints = list(checkpoint_dir.glob(f'fsbo_{args.algorithm}_*.pt'))
        if not checkpoints:
            print(f"❌ No se encontraron checkpoints para {args.algorithm}")
            return
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    print(f"\n📂 Cargando checkpoint: {checkpoint_path.name}")
    
    input_dim = X_pool.shape[1]
    model, likelihood, config = load_model_from_checkpoint(
        str(checkpoint_path), input_dim, device
    )
    
    print(f"   ✅ Modelo cargado (hidden_dim={config.get('hidden_dim', 128)})")
    
    # =========================================================================
    # Ejecutar BO
    # =========================================================================
    print("\n" + "=" * 60)
    print("🔄 Iniciando Bayesian Optimization...")
    print("=" * 60)
    
    results = run_bo_loop(
        model=model,
        likelihood=likelihood,
        X_pool=X_pool,
        y_pool=y_pool,
        n_trials=args.n_trials,
        n_init=args.n_init,
        finetune_frequency=args.finetune_freq,
        finetune_epochs=20,
        n_candidates=min(1000, len(X_pool)),
        xi=0.01,
        device=device,
        verbose=True
    )
    
    # =========================================================================
    # Resultados
    # =========================================================================
    print("\n" + "=" * 60)
    print("📋 RESULTADOS")
    print("=" * 60)
    
    print(f"\n✅ Mejor configuración encontrada:")
    print(f"   Accuracy: {results['best_y']:.4f}")
    print(f"   Óptimo real: {results['y_pool_best']:.4f}")
    print(f"   Regret: {results['regret']:.4f}")
    print(f"   Evaluaciones: {results['n_evaluations']}")
    
    # Normalized regret (Paper Sección 4.1)
    y_min = y_pool.min()
    y_max = y_pool.max()
    if y_max - y_min > 1e-8:
        normalized_regret = (y_max - results['best_y']) / (y_max - y_min)
    else:
        normalized_regret = 0.0
    
    print(f"   Normalized Regret: {normalized_regret:.4f}")
    
    # Guardar resultados
    results_file = results_dir / f"bo_results_{args.algorithm}_task{target_task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results_to_save = {
        'algorithm': args.algorithm,
        'target_task': int(target_task),
        'n_trials': args.n_trials,
        'n_init': args.n_init,
        'best_y': float(results['best_y']),
        'y_pool_best': float(results['y_pool_best']),
        'regret': float(results['regret']),
        'normalized_regret': float(normalized_regret),
        'best_y_history': [float(y) for y in results['best_y_history']],
        'checkpoint': str(checkpoint_path.name),
        'seed': args.seed
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n📁 Resultados guardados en: {results_file.name}")
    
    # Mostrar progreso
    print(f"\n📈 Progreso del mejor valor:")
    history = results['best_y_history']
    checkpoints_to_show = [0, len(history)//4, len(history)//2, 3*len(history)//4, len(history)-1]
    for i in checkpoints_to_show:
        if i < len(history):
            print(f"   Trial {i+1:3d}: {history[i]:.4f}")
    
    print("\n✅ BO completado!")


if __name__ == "__main__":
    main()

