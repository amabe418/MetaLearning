"""
Script para generar ranking de algoritmos usando Portfolio Selection.

Este script implementa el enfoque de portfolio selection basado en meta-learning:
1. Lee los k vecinos más cercanos a un dataset target (desde neighbors.csv)
2. Construye una matriz de performance (datasets × flows) desde cc18_performance.csv
3. Aplica portfolio_selection de amltk para seleccionar los mejores flows
4. Guarda el ranking en un archivo CSV

Uso:
    python benchmark/portfolio_ranking.py --neighbors-csv benchmark/neighbors.csv \\
                                          --performance-csv data/processed/cc18_performance.csv \\
                                          --output-csv benchmark/ranking.csv \\
                                          --top-k 5
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from amltk.metalearning import portfolio_selection


def build_portfolio_matrix(
    neighbor_ids: list[str],
    performance_df: pd.DataFrame,
    fillna_value: float = 0.0
) -> pd.DataFrame:
    """
    Construye la matriz de portfolio (datasets × flows) desde la tabla de performance.
    
    Args:
        neighbor_ids: Lista de openml_task de datasets vecinos
        performance_df: DataFrame con columnas [openml_task, flow, area_under_roc_curve]
        fillna_value: Valor para rellenar datos faltantes (default: 0.0)
    
    Returns:
        DataFrame con index=openml_task, columns=flow, values=area_under_roc_curve
    """
    # Filtrar solo los datasets vecinos
    portfolio_data = performance_df[performance_df['openml_task'].isin(neighbor_ids)]
    
    if portfolio_data.empty:
        raise ValueError(f"No se encontraron datos de performance para los vecinos: {neighbor_ids}")
    
    # Crear matriz pivotada: datasets en filas, flows en columnas
    portfolio = portfolio_data.pivot_table(
        index='openml_task',
        columns='flow',
        values='area_under_roc_curve',
        aggfunc='max'  # Si hay duplicados, tomar el máximo
    )
    
    # Rellenar valores faltantes
    portfolio = portfolio.fillna(fillna_value)
    
    print(portfolio)
    
    return portfolio


def compute_ranking(
    portfolio: pd.DataFrame,
    top_k: int = 5,
    scaler: Optional[str] = 'minmax',
    row_reducer=np.max,
    aggregator=np.mean
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Aplica portfolio selection para rankear flows.
    
    Args:
        portfolio: Matriz (datasets × flows) con accuracies
        top_k: Número de flows a seleccionar
        scaler: Tipo de escalado ('minmax', 'standard', None)
        row_reducer: Función para reducir por fila (default: np.max)
        aggregator: Función para agregar resultados (default: np.mean)
    
    Returns:
        Tupla (selected_portfolio, trajectory)
        - selected_portfolio: Submatriz con los k flows seleccionados
        - trajectory: Serie con flow_name → utility_score
    """
    selected_portfolio, trajectory = portfolio_selection(
        portfolio,
        k=top_k,
        scaler=scaler,
        row_reducer=row_reducer,
        aggregator=aggregator
    )
    
    return selected_portfolio, trajectory


def save_ranking(
    trajectory: pd.Series,
    output_path: Path,
    target_dataset_id: str
) -> None:
    """
    Guarda el ranking en un archivo CSV.
    
    Args:
        trajectory: Serie con flow → utility_score (ordenada)
        output_path: Ruta del archivo CSV de salida
        target_dataset_id: openml_task del dataset target
    """
    # Convertir trajectory a DataFrame con ranking
    ranking_df = pd.DataFrame({
        'rank': range(1, len(trajectory) + 1),
        'flow': trajectory.index,
        'utility_score': trajectory.values,
        'target_dataset': target_dataset_id
    })
    
    # Crear directorio si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar CSV
    ranking_df.to_csv(output_path, index=False)
    
    print(f"✓ Ranking guardado en: {output_path}")
    print(f"  Total flows rankeados: {len(ranking_df)}")


def load_neighbors(neighbors_csv: Path) -> tuple[str, list[str]]:
    """
    Carga el archivo de vecinos y extrae el target dataset y sus vecinos.
    
    Args:
        neighbors_csv: Ruta al archivo neighbors.csv
    
    Returns:
        Tupla (target_openml_task, neighbor_openml_tasks)
    """
    neighbors_df = pd.read_csv(neighbors_csv)
    
    if neighbors_df.empty:
        raise ValueError(f"El archivo {neighbors_csv} está vacío")
    
    # Verificar columnas requeridas
    required_cols = {'target_dataset', 'neighbor_id'}
    if not required_cols.issubset(neighbors_df.columns):
        raise ValueError(f"El archivo debe contener columnas: {required_cols}")
    
    # Extraer target dataset (asumiendo que todos tienen el mismo target)
    target_dataset_id = str(neighbors_df['target_dataset'].iloc[0])
    
    # Extraer IDs de vecinos (ordenados por rank si existe)
    if 'rank' in neighbors_df.columns:
        neighbors_df = neighbors_df.sort_values('rank')
    
    neighbor_ids = neighbors_df['neighbor_id'].astype(str).tolist()
    
    return target_dataset_id, neighbor_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera ranking de flows usando Portfolio Selection basado en vecinos similares"
    )
    
    parser.add_argument(
        '--neighbors-csv',
        type=str,
        required=True,
        help='Ruta al archivo CSV con los vecinos (debe tener: target_dataset, neighbor_id)'
    )
    
    parser.add_argument(
        '--performance-csv',
        type=str,
        required=True,
        help='Ruta al archivo CSV con performance (debe tener: openml_task, flow, area_under_roc_curve)'
    )
    
    parser.add_argument(
        '--output-csv',
        type=str,
        required=True,
        help='Ruta del archivo CSV de salida para el ranking'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Número de flows a incluir en el ranking (default: 5)'
    )
    
    parser.add_argument(
        '--scaler',
        type=str,
        default='minmax',
        choices=['minmax', 'standard', 'robust', 'none'],
        help='Tipo de escalado a aplicar (default: minmax)'
    )
    
    parser.add_argument(
        '--fillna',
        type=float,
        default=0.0,
        help='Valor para rellenar datos faltantes (default: 0.0)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar información detallada del proceso'
    )
    
    args = parser.parse_args()
    
    # Convertir paths
    neighbors_path = Path(args.neighbors_csv)
    performance_path = Path(args.performance_csv)
    output_path = Path(args.output_csv)
    
    # Validar archivos de entrada
    if not neighbors_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {neighbors_path}")
    if not performance_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {performance_path}")
    
    print(f"\n{'='*60}")
    print(f"Portfolio Selection - Ranking de Flows")
    print(f"{'='*60}\n")
    
    # 1. Cargar vecinos
    print(f"[1/4] Cargando vecinos desde: {neighbors_path}")
    target_dataset_id, neighbor_ids = load_neighbors(neighbors_path)
    print(f"  ✓ Target dataset: {target_dataset_id}")
    print(f"  ✓ Vecinos encontrados: {len(neighbor_ids)}")
    if args.verbose:
        print(f"    IDs: {neighbor_ids}")
    
    # 2. Construir portfolio matrix
    print(f"\n[2/4] Construyendo matriz de portfolio...")
    performance_df = pd.read_csv(performance_path)
    print(f"  ✓ Performance data cargado: {len(performance_df)} filas")
    
    scaler_arg = None if args.scaler == 'none' else args.scaler
    portfolio = build_portfolio_matrix(neighbor_ids, performance_df, args.fillna)
    print(f"  ✓ Portfolio matrix: {portfolio.shape[0]} datasets × {portfolio.shape[1]} flows")
    
    if args.verbose:
        print(f"\n  Estadísticas del portfolio:")
        print(f"    - Accuracy promedio: {portfolio.values.mean():.4f}")
        print(f"    - Accuracy máxima: {portfolio.values.max():.4f}")
        print(f"    - Valores = 0: {(portfolio.values == 0).sum()} de {portfolio.size}")
    
    # 3. Aplicar portfolio selection
    print(f"\n[3/4] Aplicando portfolio selection (k={args.top_k})...")
    selected_portfolio, trajectory = compute_ranking(
        portfolio,
        top_k=args.top_k,
        scaler=scaler_arg,
        row_reducer=np.max,
        aggregator=np.mean
    )
    print(f"  ✓ Top {args.top_k} flows seleccionados")
    
    if args.verbose:
        print(f"\n  Ranking:")
        for rank, (flow, score) in enumerate(trajectory.items(), 1):
            print(f"    {rank}. {flow:40s} → {score:.4f}")
    
    # 4. Guardar ranking
    print(f"\n[4/4] Guardando ranking...")
    save_ranking(trajectory, output_path, target_dataset_id)
    
    print(f"\n{'='*60}")
    print(f"✓ Proceso completado exitosamente")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
