"""
Métricas de evaluación para experimentos de HPO.

Implementa las métricas estándar del paper FSBO y literatura de HPO:
- Normalized Regret
- Simple Regret
- Area Under Curve (AUC)
- Convergence Speed

Referencias:
- Wistuba & Grabocka (2021) - FSBO Paper
- Eggensperger et al. (2013) - Towards an Empirical Foundation for HPO

Autor: Proyecto académico MetaLearning
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class ExperimentMetrics:
    """Contenedor para métricas de un experimento."""
    normalized_regret: float
    simple_regret: float
    best_found: float
    optimal_value: float
    n_evaluations: int
    auc: float
    convergence_curve: List[float]
    time_to_95_optimal: Optional[int]  # Evaluaciones para 95% del óptimo
    

def normalized_regret(
    y_best: float,
    y_optimal: float,
    y_worst: float
) -> float:
    """
    Calcula el Normalized Regret.
    
    Paper FSBO Sección 4.1:
        "We report the normalized regret, which is the regret divided by
        the difference between the best and worst observed performance"
    
    NR = (y* - y_best) / (y* - y_worst)
    
    Args:
        y_best: Mejor valor encontrado por el optimizador
        y_optimal: Mejor valor posible (óptimo verdadero)
        y_worst: Peor valor en el espacio de búsqueda
        
    Returns:
        Normalized Regret ∈ [0, 1], donde 0 es óptimo
    """
    denominator = y_optimal - y_worst
    
    if abs(denominator) < 1e-10:
        return 0.0 if abs(y_optimal - y_best) < 1e-10 else 1.0
    
    nr = (y_optimal - y_best) / denominator
    return np.clip(nr, 0.0, 1.0)


def simple_regret(y_best: float, y_optimal: float) -> float:
    """
    Calcula el Simple Regret.
    
    SR = y* - y_best
    
    Args:
        y_best: Mejor valor encontrado
        y_optimal: Mejor valor posible
        
    Returns:
        Simple Regret ≥ 0
    """
    return max(0.0, y_optimal - y_best)


def area_under_curve(
    convergence_curve: List[float],
    y_optimal: float,
    y_worst: float,
    normalize: bool = True
) -> float:
    """
    Calcula el Area Under the Convergence Curve (AUC).
    
    Mide la eficiencia general del optimizador:
    - AUC alto = convergencia rápida y sostenida
    - AUC bajo = convergencia lenta
    
    Args:
        convergence_curve: Lista de best_y en cada paso
        y_optimal: Mejor valor posible
        y_worst: Peor valor posible
        normalize: Si normalizar a [0, 1]
        
    Returns:
        AUC (normalizado si se especifica)
    """
    if len(convergence_curve) == 0:
        return 0.0
    
    curve = np.array(convergence_curve)
    
    if normalize and abs(y_optimal - y_worst) > 1e-10:
        # Normalizar curva a [0, 1]
        curve = (curve - y_worst) / (y_optimal - y_worst)
        curve = np.clip(curve, 0, 1)
    
    # AUC usando regla del trapecio
    auc = np.trapz(curve) / len(curve)
    
    return float(auc)


def time_to_threshold(
    convergence_curve: List[float],
    y_optimal: float,
    threshold: float = 0.95
) -> Optional[int]:
    """
    Calcula el número de evaluaciones para alcanzar un umbral del óptimo.
    
    Args:
        convergence_curve: Lista de best_y en cada paso
        y_optimal: Mejor valor posible
        threshold: Fracción del óptimo (default 95%)
        
    Returns:
        Número de evaluaciones o None si no se alcanzó
    """
    target = threshold * y_optimal
    
    for i, y in enumerate(convergence_curve):
        if y >= target:
            return i + 1  # 1-indexed
    
    return None


def compute_all_metrics(
    convergence_curve: List[float],
    y_optimal: float,
    y_worst: float
) -> ExperimentMetrics:
    """
    Calcula todas las métricas para un experimento.
    
    Args:
        convergence_curve: Lista de best_y en cada paso
        y_optimal: Mejor valor posible
        y_worst: Peor valor posible
        
    Returns:
        ExperimentMetrics con todas las métricas
    """
    y_best = max(convergence_curve) if convergence_curve else y_worst
    
    return ExperimentMetrics(
        normalized_regret=normalized_regret(y_best, y_optimal, y_worst),
        simple_regret=simple_regret(y_best, y_optimal),
        best_found=y_best,
        optimal_value=y_optimal,
        n_evaluations=len(convergence_curve),
        auc=area_under_curve(convergence_curve, y_optimal, y_worst),
        convergence_curve=convergence_curve,
        time_to_95_optimal=time_to_threshold(convergence_curve, y_optimal, 0.95)
    )


# =============================================================================
# Agregación de métricas sobre múltiples experimentos
# =============================================================================

@dataclass
class AggregatedMetrics:
    """Métricas agregadas sobre múltiples repeticiones."""
    method_name: str
    n_experiments: int
    
    # Normalized Regret
    nr_mean: float
    nr_std: float
    nr_median: float
    
    # Simple Regret
    sr_mean: float
    sr_std: float
    
    # AUC
    auc_mean: float
    auc_std: float
    
    # Convergence
    convergence_mean: np.ndarray
    convergence_std: np.ndarray
    
    # Time to threshold
    time_to_95_mean: Optional[float]
    time_to_95_std: Optional[float]
    
    # Raw data for statistical tests
    all_normalized_regrets: List[float]
    all_aucs: List[float]


def aggregate_metrics(
    experiments: List[ExperimentMetrics],
    method_name: str = "unknown"
) -> AggregatedMetrics:
    """
    Agrega métricas de múltiples experimentos.
    
    Args:
        experiments: Lista de ExperimentMetrics
        method_name: Nombre del método
        
    Returns:
        AggregatedMetrics con media ± std
    """
    n = len(experiments)
    
    if n == 0:
        raise ValueError("No experiments to aggregate")
    
    # Extraer valores
    nrs = [e.normalized_regret for e in experiments]
    srs = [e.simple_regret for e in experiments]
    aucs = [e.auc for e in experiments]
    times = [e.time_to_95_optimal for e in experiments if e.time_to_95_optimal is not None]
    
    # Agregar curvas de convergencia
    max_len = max(len(e.convergence_curve) for e in experiments)
    curves = np.zeros((n, max_len))
    
    for i, e in enumerate(experiments):
        curve = e.convergence_curve
        # Pad con el último valor si es necesario
        if len(curve) < max_len:
            curve = curve + [curve[-1]] * (max_len - len(curve))
        curves[i] = curve[:max_len]
    
    return AggregatedMetrics(
        method_name=method_name,
        n_experiments=n,
        nr_mean=np.mean(nrs),
        nr_std=np.std(nrs),
        nr_median=np.median(nrs),
        sr_mean=np.mean(srs),
        sr_std=np.std(srs),
        auc_mean=np.mean(aucs),
        auc_std=np.std(aucs),
        convergence_mean=np.mean(curves, axis=0),
        convergence_std=np.std(curves, axis=0),
        time_to_95_mean=np.mean(times) if times else None,
        time_to_95_std=np.std(times) if times else None,
        all_normalized_regrets=nrs,
        all_aucs=aucs
    )


# =============================================================================
# Tests estadísticos
# =============================================================================

def wilcoxon_test(
    method1_values: List[float],
    method2_values: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test para comparar dos métodos.
    
    Paper FSBO usa este test para comparaciones pareadas.
    
    Args:
        method1_values: Valores del método 1 (ej: normalized regrets)
        method2_values: Valores del método 2
        alternative: 'two-sided', 'less', 'greater'
        
    Returns:
        (statistic, p_value)
    """
    try:
        stat, p_value = stats.wilcoxon(
            method1_values, 
            method2_values,
            alternative=alternative
        )
        return float(stat), float(p_value)
    except ValueError:
        # Si todos los valores son iguales
        return 0.0, 1.0


def mann_whitney_test(
    method1_values: List[float],
    method2_values: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Mann-Whitney U test para comparar dos métodos (no pareado).
    
    Args:
        method1_values: Valores del método 1
        method2_values: Valores del método 2
        alternative: 'two-sided', 'less', 'greater'
        
    Returns:
        (statistic, p_value)
    """
    stat, p_value = stats.mannwhitneyu(
        method1_values,
        method2_values,
        alternative=alternative
    )
    return float(stat), float(p_value)


def compare_methods(
    method1: AggregatedMetrics,
    method2: AggregatedMetrics,
    alpha: float = 0.05
) -> Dict:
    """
    Compara dos métodos estadísticamente.
    
    Args:
        method1: Métricas agregadas del método 1
        method2: Métricas agregadas del método 2
        alpha: Nivel de significancia
        
    Returns:
        Dict con resultados de comparación
    """
    # Wilcoxon test en normalized regret
    stat_nr, p_nr = wilcoxon_test(
        method1.all_normalized_regrets,
        method2.all_normalized_regrets
    )
    
    # Wilcoxon test en AUC
    stat_auc, p_auc = wilcoxon_test(
        method1.all_aucs,
        method2.all_aucs
    )
    
    # Determinar ganador
    if p_nr < alpha:
        if method1.nr_mean < method2.nr_mean:
            nr_winner = method1.method_name
        else:
            nr_winner = method2.method_name
    else:
        nr_winner = "tie"
    
    if p_auc < alpha:
        if method1.auc_mean > method2.auc_mean:
            auc_winner = method1.method_name
        else:
            auc_winner = method2.method_name
    else:
        auc_winner = "tie"
    
    return {
        'method1': method1.method_name,
        'method2': method2.method_name,
        'normalized_regret': {
            'method1_mean': method1.nr_mean,
            'method2_mean': method2.nr_mean,
            'statistic': stat_nr,
            'p_value': p_nr,
            'significant': bool(p_nr < alpha),
            'winner': nr_winner
        },
        'auc': {
            'method1_mean': method1.auc_mean,
            'method2_mean': method2.auc_mean,
            'statistic': stat_auc,
            'p_value': p_auc,
            'significant': bool(p_auc < alpha),
            'winner': auc_winner
        }
    }


# =============================================================================
# Utilidades para reporting
# =============================================================================

def format_metric(mean: float, std: float, precision: int = 4) -> str:
    """Formatea métrica como 'mean ± std'."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def metrics_to_table_row(metrics: AggregatedMetrics) -> Dict:
    """Convierte métricas a fila de tabla."""
    return {
        'Method': metrics.method_name,
        'NR': format_metric(metrics.nr_mean, metrics.nr_std),
        'SR': format_metric(metrics.sr_mean, metrics.sr_std),
        'AUC': format_metric(metrics.auc_mean, metrics.auc_std),
        'Time to 95%': f"{metrics.time_to_95_mean:.1f}" if metrics.time_to_95_mean else "N/A",
        'N': metrics.n_experiments
    }


def print_comparison_table(methods: List[AggregatedMetrics]):
    """Imprime tabla de comparación de métodos."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<20} {'NR (↓)':<20} {'AUC (↑)':<20} {'Time to 95%':<15}")
    print("-" * 80)
    
    # Ordenar por NR (menor es mejor)
    sorted_methods = sorted(methods, key=lambda m: m.nr_mean)
    
    for m in sorted_methods:
        nr_str = format_metric(m.nr_mean, m.nr_std)
        auc_str = format_metric(m.auc_mean, m.auc_std)
        time_str = f"{m.time_to_95_mean:.1f}" if m.time_to_95_mean else "N/A"
        print(f"{m.method_name:<20} {nr_str:<20} {auc_str:<20} {time_str:<15}")
    
    print("=" * 80)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing metrics module...")
    
    # Datos de prueba
    y_optimal = 0.95
    y_worst = 0.50
    
    curve1 = [0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93]
    curve2 = [0.52, 0.55, 0.58, 0.62, 0.68, 0.72, 0.75, 0.78, 0.80, 0.82]
    
    # Calcular métricas
    m1 = compute_all_metrics(curve1, y_optimal, y_worst)
    m2 = compute_all_metrics(curve2, y_optimal, y_worst)
    
    print(f"\nMethod 1:")
    print(f"  NR: {m1.normalized_regret:.4f}")
    print(f"  SR: {m1.simple_regret:.4f}")
    print(f"  AUC: {m1.auc:.4f}")
    print(f"  Time to 95%: {m1.time_to_95_optimal}")
    
    print(f"\nMethod 2:")
    print(f"  NR: {m2.normalized_regret:.4f}")
    print(f"  SR: {m2.simple_regret:.4f}")
    print(f"  AUC: {m2.auc:.4f}")
    print(f"  Time to 95%: {m2.time_to_95_optimal}")
    
    # Test agregación
    experiments1 = [m1] * 5  # Simular 5 repeticiones
    experiments2 = [m2] * 5
    
    agg1 = aggregate_metrics(experiments1, "FSBO")
    agg2 = aggregate_metrics(experiments2, "Random")
    
    print_comparison_table([agg1, agg2])
    
    # Test estadístico
    comparison = compare_methods(agg1, agg2)
    print(f"\nStatistical comparison:")
    print(f"  NR winner: {comparison['normalized_regret']['winner']}")
    print(f"  AUC winner: {comparison['auc']['winner']}")
    
    print("\n✅ Metrics module tests passed!")

