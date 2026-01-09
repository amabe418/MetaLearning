"""
Framework de experimentación para FSBO con K-Fold Cross-Validation.

Implementa evaluación rigurosa usando:
1. K-Fold Cross-Validation sobre TAREAS (no muestras)
2. Múltiples semillas aleatorias por fold
3. Tests estadísticos (Wilcoxon, Friedman)
4. Comparación con baselines

Protocolo experimental:
- Divide las N tareas en K folds
- Para cada fold k:
    - Train: tareas de folds ≠ k (para meta-training)
    - Test: tareas del fold k
- Cada tarea aparece exactamente UNA vez en test
- Repetir con múltiples seeds para varianza

Referencias:
- Wistuba & Grabocka (2021) - FSBO Paper
- Japkowicz & Shah (2011) - Evaluating Learning Algorithms

Uso:
    python scripts/experiments.py --algorithm adaboost --k_folds 5 --n_seeds 5
    python scripts/experiments.py --algorithm all --k_folds 5 --n_seeds 10

Autor: Proyecto académico MetaLearning
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

# Imports locales
from metrics import (
    compute_all_metrics,
    aggregate_metrics,
    compare_methods,
    print_comparison_table,
    ExperimentMetrics,
    AggregatedMetrics
)
from baselines import run_baseline, OptimizationTrace
from fsbo_optimizer import FSBOOptimizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuración de algoritmos
# =============================================================================

ALGORITHM_FILES = {
    'adaboost': 'adaboost_target_representation_with_scores.csv',
    'random_forest': 'random_forest_target_representation_with_scores.csv',
    'libsvm_svc': 'libsvm_svc_target_representation_with_scores.csv',
    'autosklearn': 'autosklearn_target_representation_with_scores.csv',
}


# =============================================================================
# Carga de datos
# =============================================================================

def load_task_data(
    algorithm: str,
    data_dir: Path
) -> Dict[int, Dict]:
    """
    Carga datos organizados por tarea.
    
    Returns:
        Dict[task_id, {'X': array, 'y': array, 'y_optimal': float, 'y_worst': float}]
    """
    file_path = data_dir / ALGORITHM_FILES[algorithm]
    df = pd.read_csv(file_path)
    
    task_col = 'task_id'
    score_col = 'accuracy'
    hp_cols = [c for c in df.columns if c not in [task_col, score_col]]
    
    tasks = {}
    
    for task_id, group in df.groupby(task_col):
        X = group[hp_cols].values.astype(np.float32)
        y = group[score_col].values.astype(np.float32)
        
        tasks[int(task_id)] = {
            'X': X,
            'y': y,
            'y_optimal': float(y.max()),
            'y_worst': float(y.min()),
            'n_samples': len(y)
        }
    
    return tasks


# =============================================================================
# K-Fold Cross-Validation sobre Tareas
# =============================================================================

def create_task_kfold(
    task_ids: List[int],
    k_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Crea K-Fold sobre TAREAS (no muestras).
    
    En meta-learning, la división debe ser sobre tareas para evaluar
    generalización a tareas nuevas (nunca vistas durante entrenamiento).
    
    Args:
        task_ids: Lista de IDs de tareas
        k_folds: Número de folds
        shuffle: Si mezclar antes de dividir
        random_state: Semilla para reproducibilidad
        
    Returns:
        Lista de (train_task_ids, test_task_ids) para cada fold
    """
    kf = KFold(n_splits=k_folds, shuffle=shuffle, random_state=random_state)
    
    task_ids_array = np.array(task_ids)
    folds = []
    
    for train_idx, test_idx in kf.split(task_ids_array):
        train_tasks = task_ids_array[train_idx].tolist()
        test_tasks = task_ids_array[test_idx].tolist()
        folds.append((train_tasks, test_tasks))
    
    return folds


# =============================================================================
# Evaluación en una tarea
# =============================================================================

def create_evaluation_function(task_data: Dict) -> callable:
    """
    Crea función de evaluación para una tarea.
    
    Simula evaluación buscando en el pool de configuraciones.
    """
    X_pool = task_data['X']
    y_pool = task_data['y']
    
    def evaluate(x: np.ndarray) -> float:
        """Encuentra la configuración más cercana y retorna su score."""
        distances = np.linalg.norm(X_pool - x, axis=1)
        closest_idx = np.argmin(distances)
        return float(y_pool[closest_idx])
    
    return evaluate


def run_fsbo_on_task(
    task_data: Dict,
    optimizer: FSBOOptimizer,
    n_trials: int = 50,
    n_init: int = 5,
    verbose: bool = False
) -> OptimizationTrace:
    """
    Ejecuta FSBO en una tarea.
    """
    evaluate_fn = create_evaluation_function(task_data)
    
    optimizer.reset()
    
    # Warm start
    initial_configs = optimizer.suggest_initial(n_init)
    
    for config in initial_configs:
        x = np.array([config[k] for k in sorted(config.keys())], dtype=np.float32)
        y = evaluate_fn(x)
        optimizer.observe(config, y)
    
    # BO loop
    for i in range(n_trials - n_init):
        config = optimizer.suggest()
        x = np.array([config[k] for k in sorted(config.keys())], dtype=np.float32)
        y = evaluate_fn(x)
        optimizer.observe(config, y)
        
        if verbose and (i + 1) % 10 == 0:
            best = max(optimizer.y_observed)
            print(f"  [FSBO] Trial {n_init + i + 1}/{n_trials}: best={best:.4f}")
    
    return OptimizationTrace(
        method="FSBO",
        observed_x=[np.array([c[k] for k in sorted(c.keys())]) for c in optimizer.configs_observed],
        observed_y=optimizer.y_observed.copy(),
        best_y_history=optimizer.best_y_history.copy()
    )


def run_baseline_on_task(
    task_data: Dict,
    method: str,
    n_trials: int = 50,
    n_init: int = 5,
    seed: int = 42,
    verbose: bool = False
) -> OptimizationTrace:
    """
    Ejecuta un baseline en una tarea.
    """
    evaluate_fn = create_evaluation_function(task_data)
    input_dim = task_data['X'].shape[1]
    
    return run_baseline(
        method=method,
        evaluate_fn=evaluate_fn,
        input_dim=input_dim,
        n_trials=n_trials,
        n_init=n_init,
        seed=seed,
        verbose=verbose
    )


# =============================================================================
# Experimento con K-Fold Cross-Validation
# =============================================================================

def run_kfold_experiment(
    algorithm: str,
    tasks: Dict[int, Dict],
    checkpoint_dir: Path,
    k_folds: int = 5,
    n_trials: int = 50,
    n_init: int = 5,
    n_seeds: int = 5,
    methods: List[str] = ['fsbo', 'random', 'gp-lhs', 'gp-rs'],
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Ejecuta experimento K-Fold completo.
    
    Protocolo:
    1. Dividir tareas en K folds
    2. Para cada fold k:
        - Test en tareas del fold k
        - Para cada tarea de test:
            - Para cada seed:
                - Ejecutar todos los métodos
    3. Agregar resultados
    
    Args:
        algorithm: Algoritmo a evaluar
        tasks: Datos de todas las tareas
        checkpoint_dir: Directorio de checkpoints FSBO
        k_folds: Número de folds
        n_trials: Presupuesto de evaluaciones
        n_init: Configuraciones iniciales
        n_seeds: Repeticiones por tarea
        methods: Métodos a comparar
        random_state: Semilla global
        verbose: Mostrar progreso
        
    Returns:
        Dict con resultados agregados y por fold
    """
    task_ids = list(tasks.keys())
    folds = create_task_kfold(task_ids, k_folds, shuffle=True, random_state=random_state)
    
    # Cargar FSBO si está disponible
    fsbo_optimizer = None
    if 'fsbo' in methods:
        try:
            fsbo_optimizer = FSBOOptimizer.from_pretrained(algorithm, str(checkpoint_dir))
        except FileNotFoundError:
            logger.warning(f"No FSBO checkpoint for {algorithm}, skipping FSBO")
            methods = [m for m in methods if m != 'fsbo']
    
    # Almacenar resultados
    all_results = {method: [] for method in methods}
    fold_results = []
    
    total_tasks = sum(len(test_ids) for _, test_ids in folds)
    total_experiments = total_tasks * n_seeds * len(methods)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"K-FOLD CROSS-VALIDATION EXPERIMENT: {algorithm.upper()}")
        print(f"{'='*70}")
        print(f"Total tasks: {len(task_ids)}")
        print(f"K-Folds: {k_folds}")
        print(f"Seeds per task: {n_seeds}")
        print(f"Methods: {methods}")
        print(f"Budget: {n_trials} evaluations")
        print(f"Total experiments: {total_experiments}")
    
    pbar = tqdm(total=total_experiments, desc="K-Fold Experiments", disable=not verbose)
    
    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        if verbose:
            tqdm.write(f"\n{'─'*50}")
            tqdm.write(f"FOLD {fold_idx + 1}/{k_folds}")
            tqdm.write(f"  Train tasks: {len(train_ids)}")
            tqdm.write(f"  Test tasks: {len(test_ids)}")
            tqdm.write(f"{'─'*50}")
        
        fold_method_results = {method: [] for method in methods}
        
        for task_id in test_ids:
            task_data = tasks[task_id]
            y_optimal = task_data['y_optimal']
            y_worst = task_data['y_worst']
            
            for seed in range(n_seeds):
                np.random.seed(random_state * 1000 + fold_idx * 100 + seed)
                torch.manual_seed(random_state * 1000 + fold_idx * 100 + seed)
                
                for method in methods:
                    if method == 'fsbo':
                        trace = run_fsbo_on_task(
                            task_data=task_data,
                            optimizer=fsbo_optimizer,
                            n_trials=n_trials,
                            n_init=n_init,
                            verbose=False
                        )
                    else:
                        trace = run_baseline_on_task(
                            task_data=task_data,
                            method=method,
                            n_trials=n_trials,
                            n_init=n_init,
                            seed=seed,
                            verbose=False
                        )
                    
                    # Calcular métricas
                    metrics = compute_all_metrics(
                        convergence_curve=trace.best_y_history,
                        y_optimal=y_optimal,
                        y_worst=y_worst
                    )
                    
                    all_results[method].append(metrics)
                    fold_method_results[method].append(metrics)
                    pbar.update(1)
        
        # Agregar resultados del fold
        fold_aggregated = {}
        for method, experiments in fold_method_results.items():
            if experiments:
                fold_aggregated[method] = aggregate_metrics(experiments, method.upper())
        
        fold_results.append({
            'fold': fold_idx + 1,
            'train_tasks': train_ids,
            'test_tasks': test_ids,
            'results': fold_aggregated
        })
        
        # Mostrar resultados del fold
        if verbose:
            tqdm.write(f"\n  Fold {fold_idx + 1} Results:")
            for method, agg in fold_aggregated.items():
                tqdm.write(f"    {method}: NR={agg.nr_mean:.4f}±{agg.nr_std:.4f}, AUC={agg.auc_mean:.4f}")
    
    pbar.close()
    
    # Agregar resultados globales
    global_aggregated = {}
    for method, experiments in all_results.items():
        if experiments:
            global_aggregated[method] = aggregate_metrics(experiments, method.upper())
    
    return {
        'algorithm': algorithm,
        'k_folds': k_folds,
        'n_seeds': n_seeds,
        'n_trials': n_trials,
        'n_init': n_init,
        'methods': methods,
        'n_total_tasks': len(task_ids),
        'fold_results': fold_results,
        'global_results': global_aggregated,
        'raw_results': all_results  # Para tests estadísticos
    }


# =============================================================================
# Tests estadísticos avanzados
# =============================================================================

def friedman_test(results: Dict[str, List[ExperimentMetrics]]) -> Dict:
    """
    Friedman test para comparar múltiples métodos.
    
    Usado cuando tenemos más de 2 métodos para comparar.
    Similar a ANOVA no paramétrico.
    """
    from scipy import stats
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    if n_methods < 3:
        return {'error': 'Friedman test requires at least 3 methods'}
    
    # Extraer NR de cada experimento
    # Asumimos que los experimentos están alineados (mismo orden de tareas)
    n_experiments = min(len(results[m]) for m in methods)
    
    # Crear matriz de datos
    data = np.zeros((n_experiments, n_methods))
    for j, method in enumerate(methods):
        data[:, j] = [results[method][i].normalized_regret for i in range(n_experiments)]
    
    # Calcular rankings por fila (experimento)
    rankings = np.zeros_like(data)
    for i in range(n_experiments):
        rankings[i] = stats.rankdata(data[i])  # Menor es mejor para NR
    
    # Friedman test
    statistic, p_value = stats.friedmanchisquare(*[rankings[:, j] for j in range(n_methods)])
    
    # Average ranks
    avg_ranks = {method: float(rankings[:, j].mean()) for j, method in enumerate(methods)}
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),  # Convert to Python bool
        'average_ranks': avg_ranks,
        'n_experiments': n_experiments
    }


def nemenyi_post_hoc(results: Dict[str, List[ExperimentMetrics]], alpha: float = 0.05) -> Dict:
    """
    Nemenyi post-hoc test después de Friedman.
    
    Determina qué pares de métodos son significativamente diferentes.
    """
    from scipy import stats
    
    methods = list(results.keys())
    n_methods = len(methods)
    n_experiments = min(len(results[m]) for m in methods)
    
    # Calcular rankings
    data = np.zeros((n_experiments, n_methods))
    for j, method in enumerate(methods):
        data[:, j] = [results[method][i].normalized_regret for i in range(n_experiments)]
    
    rankings = np.zeros_like(data)
    for i in range(n_experiments):
        rankings[i] = stats.rankdata(data[i])
    
    avg_ranks = rankings.mean(axis=0)
    
    # Critical difference (Nemenyi)
    q_alpha = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    
    if n_methods > 10:
        q = 3.2  # Approximate
    else:
        q = q_alpha.get(n_methods, 2.5)
    
    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6 * n_experiments))
    
    # Comparaciones pareadas
    comparisons = []
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                diff = abs(avg_ranks[i] - avg_ranks[j])
                significant = diff > cd
                comparisons.append({
                    'method1': m1,
                    'method2': m2,
                    'rank_diff': float(diff),
                    'critical_diff': float(cd),
                    'significant': bool(significant)  # Convert to Python bool
                })
    
    return {
        'critical_difference': float(cd),
        'average_ranks': {m: float(avg_ranks[i]) for i, m in enumerate(methods)},
        'comparisons': comparisons
    }


# =============================================================================
# Guardar resultados
# =============================================================================

def save_kfold_results(
    results: Dict,
    output_dir: Path
) -> Path:
    """
    Guarda resultados del experimento K-Fold.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    algorithm = results['algorithm']
    k_folds = results['k_folds']
    filename = f"kfold_{algorithm}_{k_folds}fold_{timestamp}.json"
    filepath = output_dir / filename
    
    # Preparar datos para guardar
    data = {
        'algorithm': algorithm,
        'timestamp': timestamp,
        'config': {
            'k_folds': k_folds,
            'n_seeds': results['n_seeds'],
            'n_trials': results['n_trials'],
            'n_init': results['n_init'],
            'methods': results['methods'],
            'n_total_tasks': results['n_total_tasks']
        },
        'fold_results': [],
        'global_results': {},
        'statistical_tests': {}
    }
    
    # Resultados por fold
    for fold in results['fold_results']:
        fold_data = {
            'fold': fold['fold'],
            'train_tasks': fold['train_tasks'],
            'test_tasks': fold['test_tasks'],
            'results': {}
        }
        for method, metrics in fold['results'].items():
            fold_data['results'][method] = {
                'normalized_regret': {'mean': float(metrics.nr_mean), 'std': float(metrics.nr_std)},
                'auc': {'mean': float(metrics.auc_mean), 'std': float(metrics.auc_std)}
            }
        data['fold_results'].append(fold_data)
    
    # Resultados globales
    for method, metrics in results['global_results'].items():
        data['global_results'][method] = {
            'normalized_regret': {
                'mean': float(metrics.nr_mean),
                'std': float(metrics.nr_std),
                'median': float(metrics.nr_median)
            },
            'simple_regret': {
                'mean': float(metrics.sr_mean),
                'std': float(metrics.sr_std)
            },
            'auc': {
                'mean': float(metrics.auc_mean),
                'std': float(metrics.auc_std)
            },
            'time_to_95': {
                'mean': float(metrics.time_to_95_mean) if metrics.time_to_95_mean else None,
                'std': float(metrics.time_to_95_std) if metrics.time_to_95_std else None
            },
            'n_experiments': metrics.n_experiments,
            'convergence_mean': metrics.convergence_mean.tolist(),
            'convergence_std': metrics.convergence_std.tolist()
        }
    
    # Tests estadísticos
    if len(results['methods']) >= 3:
        data['statistical_tests']['friedman'] = friedman_test(results['raw_results'])
        data['statistical_tests']['nemenyi'] = nemenyi_post_hoc(results['raw_results'])
    
    # Comparaciones pareadas (Wilcoxon)
    data['statistical_tests']['pairwise_wilcoxon'] = []
    method_list = list(results['global_results'].values())
    method_names = list(results['global_results'].keys())
    
    for i, m1 in enumerate(method_list):
        for j, m2 in enumerate(method_list[i+1:], i+1):
            comparison = compare_methods(m1, m2)
            data['statistical_tests']['pairwise_wilcoxon'].append(comparison)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath


# =============================================================================
# Reporting
# =============================================================================

def print_kfold_summary(results: Dict):
    """
    Imprime resumen de resultados K-Fold.
    """
    print("\n" + "=" * 80)
    print(f"📊 K-FOLD CROSS-VALIDATION SUMMARY: {results['algorithm'].upper()}")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  K-Folds: {results['k_folds']}")
    print(f"  Seeds per task: {results['n_seeds']}")
    print(f"  Total tasks: {results['n_total_tasks']}")
    print(f"  Budget: {results['n_trials']}")
    
    # Tabla de resultados globales
    print("\n" + "-" * 80)
    print("GLOBAL RESULTS (across all folds)")
    print("-" * 80)
    print(f"{'Method':<15} {'NR (↓)':<20} {'AUC (↑)':<20} {'Time to 95%':<15}")
    print("-" * 80)
    
    sorted_methods = sorted(
        results['global_results'].items(),
        key=lambda x: x[1].nr_mean
    )
    
    for method, metrics in sorted_methods:
        nr_str = f"{metrics.nr_mean:.4f} ± {metrics.nr_std:.4f}"
        auc_str = f"{metrics.auc_mean:.4f} ± {metrics.auc_std:.4f}"
        time_str = f"{metrics.time_to_95_mean:.1f}" if metrics.time_to_95_mean else "N/A"
        print(f"{method:<15} {nr_str:<20} {auc_str:<20} {time_str:<15}")
    
    print("-" * 80)
    
    # Resultados por fold
    print("\nRESULTS PER FOLD:")
    print("-" * 80)
    
    for fold in results['fold_results']:
        print(f"\nFold {fold['fold']}:")
        for method, metrics in fold['results'].items():
            print(f"  {method}: NR={metrics.nr_mean:.4f}±{metrics.nr_std:.4f}")
    
    # Winner
    best_method = sorted_methods[0][0]
    print("\n" + "=" * 80)
    print(f"🏆 BEST METHOD: {best_method}")
    print(f"   NR: {sorted_methods[0][1].nr_mean:.4f} ± {sorted_methods[0][1].nr_std:.4f}")
    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run FSBO K-Fold Cross-Validation Experiments')
    parser.add_argument('--algorithm', type=str, default='adaboost',
                       choices=list(ALGORITHM_FILES.keys()) + ['all'])
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Budget per task')
    parser.add_argument('--n_init', type=int, default=5,
                       help='Initial configurations')
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of random seeds per task')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['fsbo', 'random', 'gp-lhs', 'gp-rs'],
                       help='Methods to compare')
    parser.add_argument('--seed', type=int, default=42,
                       help='Global random seed')
    
    args = parser.parse_args()
    
    # Rutas
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'representation_with_scores'
    checkpoint_dir = base_dir / 'experiments' / 'checkpoints'
    results_dir = base_dir / 'experiments' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🧪 FSBO K-FOLD CROSS-VALIDATION FRAMEWORK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  K-Folds: {args.k_folds}")
    print(f"  Budget: {args.n_trials}")
    print(f"  Initial: {args.n_init}")
    print(f"  Seeds per task: {args.n_seeds}")
    print(f"  Methods: {args.methods}")
    print(f"  Global seed: {args.seed}")
    
    # Algoritmos a evaluar
    algorithms = list(ALGORITHM_FILES.keys()) if args.algorithm == 'all' else [args.algorithm]
    
    all_experiment_results = {}
    
    for algorithm in algorithms:
        print(f"\n{'#'*80}")
        print(f"# ALGORITHM: {algorithm.upper()}")
        print(f"{'#'*80}")
        
        # Cargar datos
        print("\n📂 Loading data...")
        tasks = load_task_data(algorithm, data_dir)
        print(f"   Loaded {len(tasks)} tasks")
        
        # Ejecutar experimento K-Fold
        results = run_kfold_experiment(
            algorithm=algorithm,
            tasks=tasks,
            checkpoint_dir=checkpoint_dir,
            k_folds=args.k_folds,
            n_trials=args.n_trials,
            n_init=args.n_init,
            n_seeds=args.n_seeds,
            methods=args.methods,
            random_state=args.seed,
            verbose=True
        )
        
        all_experiment_results[algorithm] = results
        
        # Imprimir resumen
        print_kfold_summary(results)
        
        # Guardar resultados
        filepath = save_kfold_results(results, results_dir)
        print(f"\n📁 Results saved to: {filepath.name}")
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📋 FINAL SUMMARY - ALL ALGORITHMS")
    print("=" * 80)
    
    for algorithm, results in all_experiment_results.items():
        global_results = results['global_results']
        best_method = min(global_results.items(), key=lambda x: x[1].nr_mean)
        
        print(f"\n{algorithm.upper()}:")
        print(f"  Best method: {best_method[0]}")
        print(f"  NR: {best_method[1].nr_mean:.4f} ± {best_method[1].nr_std:.4f}")
        
        if 'fsbo' in global_results:
            fsbo_nr = global_results['fsbo'].nr_mean
            better_than = sum(1 for m, r in global_results.items() if m != 'fsbo' and r.nr_mean > fsbo_nr)
            print(f"  FSBO beats {better_than}/{len(global_results)-1} baselines")
    
    print("\n" + "=" * 80)
    print("✅ K-Fold Cross-Validation experiments completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
