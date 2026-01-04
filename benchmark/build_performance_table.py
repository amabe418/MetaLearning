"""
Script para construir la tabla de performance (dataset, flow, accuracy) desde OpenML.

Uso:
    python benchmark/build_performance_table.py --flows data/processed/top_30_flows.json --benchmark openml_cc18
    python benchmark/build_performance_table.py --flows data/processed/top_30_flows.json --benchmark openml_100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import openml
import pandas as pd
from tqdm import tqdm

# Agregar config al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
from config import benchmarks


def get_best_accuracy_for_flow(dataset_id: int, flow_id: int) -> Optional[float]:
    """
    Obtiene el mejor accuracy para un flow en un dataset específico.
    
    Args:
        dataset_id: ID del dataset en OpenML
        flow_id: ID del flow en OpenML
        
    Returns:
        Mejor accuracy encontrado, o None si no hay runs
    """
    try:
        # Obtener el task_id del dataset
        task = openml.tasks.get_task(dataset_id)
        task_id = task.task_id
        
        # Obtener evaluaciones para este task y flow
        evaluations = openml.evaluations.list_evaluations(
            function='predictive_accuracy',
            tasks=[task_id],
            flows=[flow_id],
            output_format='dataframe'
        )
        
        if evaluations is None or evaluations.empty:
            return None
        
        # Obtener el mejor accuracy
        best_accuracy = evaluations['value'].max()
        return float(best_accuracy)
        
    except Exception as e:
        # Silenciar errores individuales para no saturar el output
        return None


def build_performance_table(
    dataset_ids: List[int],
    flows: Dict[int, str],
    default_accuracy: float = 0.0
) -> pd.DataFrame:
    """
    Construye la tabla de performance para datasets y flows.
    
    Args:
        dataset_ids: Lista de dataset IDs
        flows: Diccionario {flow_id: flow_name}
        default_accuracy: Accuracy a usar cuando no hay runs (default: 0.0)
        
    Returns:
        DataFrame con columnas [dataset_id, flow_name, accuracy]
    """
    print(f"\n{'='*60}")
    print(f"Construyendo tabla de performance...")
    print(f"  Datasets: {len(dataset_ids)}")
    print(f"  Flows: {len(flows)}")
    print(f"  Combinaciones totales: {len(dataset_ids) * len(flows)}")
    print(f"{'='*60}\n")
    
    results = []
    total_combinations = len(dataset_ids) * len(flows)
    found_count = 0
    missing_count = 0
    
    # Crear barra de progreso para todas las combinaciones
    with tqdm(total=total_combinations, desc="Procesando") as pbar:
        for dataset_id in dataset_ids:
            for flow_id, flow_name in flows.items():
                # Obtener accuracy
                accuracy = get_best_accuracy_for_flow(dataset_id, flow_id)
                
                if accuracy is None:
                    accuracy = default_accuracy
                    missing_count += 1
                else:
                    found_count += 1
                
                # Agregar resultado
                results.append({
                    'dataset_id': dataset_id,
                    'flow_id': flow_id,
                    'flow_name': flow_name,
                    'accuracy': accuracy
                })
                
                pbar.update(1)
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("Estadísticas:")
    print(f"{'='*60}")
    print(f"  Combinaciones procesadas: {total_combinations}")
    print(f"  Runs encontrados: {found_count} ({found_count/total_combinations*100:.1f}%)")
    print(f"  Runs faltantes (penalizados): {missing_count} ({missing_count/total_combinations*100:.1f}%)")
    print(f"  Accuracy promedio: {df['accuracy'].mean():.4f}")
    print(f"  Accuracy mediana: {df['accuracy'].median():.4f}")
    print(f"{'='*60}\n")
    
    return df


def get_datasets_from_benchmark(benchmark_name: str) -> List[int]:
    """
    Obtiene la lista de datasets desde la configuración de benchmarks.
    
    Args:
        benchmark_name: Nombre del benchmark en benchmarks.yaml
        
    Returns:
        Lista de dataset IDs
    """
    if benchmark_name not in benchmarks:
        print(f"Error: Benchmark '{benchmark_name}' no encontrado")
        print(f"Benchmarks disponibles: {list(benchmarks.keys())}")
        sys.exit(1)
    
    bench_cfg = benchmarks[benchmark_name]
    
    try:
        # Obtener suite de OpenML
        if "suite_id" in bench_cfg:
            suite = openml.study.get_suite(bench_cfg["suite_id"])
        elif "suite_name" in bench_cfg:
            suite = openml.study.get_suite(bench_cfg["suite_name"])
        else:
            raise ValueError("Benchmark config must have 'suite_id' or 'suite_name'")
        
        # Obtener dataset IDs desde tasks
        dataset_ids = []
        print(f"Obteniendo datasets de {len(suite.tasks)} tasks...")
        for task_id in tqdm(suite.tasks, desc="Procesando tasks"):
            try:
                task = openml.tasks.get_task(task_id)
                dataset_ids.append(task.dataset_id)
            except Exception as e:
                print(f"  Warning: Error con task {task_id}: {e}")
        
        return dataset_ids
        
    except Exception as e:
        print(f"Error obteniendo datasets del benchmark: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Construye tabla de performance desde OpenML"
    )
    
    parser.add_argument(
        '--flows',
        type=str,
        required=True,
        help='Ruta al archivo JSON con los flows seleccionados'
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='Nombre del benchmark desde config/benchmarks.yaml (ej: openml_cc18, openml_100)'
    )
    
    parser.add_argument(
        '--default-accuracy',
        type=float,
        default=0.0,
        help='Accuracy para flows no ejecutados (default: 0.0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/performance_table.csv',
        help='Ruta del archivo CSV de salida'
    )
    
    parser.add_argument(
        '--include-flow-id',
        action='store_true',
        help='Incluir columna flow_id en el CSV de salida'
    )
    
    args = parser.parse_args()
    
    # Cargar flows
    print(f"Cargando flows desde: {args.flows}")
    with open(args.flows, 'r') as f:
        flows_dict = json.load(f)
    
    # Convertir keys a int
    flows = {int(k): v for k, v in flows_dict.items()}
    print(f"  ✓ {len(flows)} flows cargados\n")
    
    # Obtener datasets del benchmark
    print(f"Usando benchmark: {args.benchmark}")
    dataset_ids = get_datasets_from_benchmark(args.benchmark)
    
    print(f"\nDatasets a procesar: {len(dataset_ids)}")
    print(f"Primeros 10: {dataset_ids[:10]}...\n")
    
    # Construir tabla
    df = build_performance_table(
        dataset_ids=dataset_ids,
        flows=flows,
        default_accuracy=args.default_accuracy
    )
    
    # Preparar salida
    if not args.include_flow_id:
        df = df[['dataset_id', 'flow_name', 'accuracy']]
    
    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"✓ Tabla guardada en: {output_path}")
    print(f"  Forma: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    # Mostrar muestra
    print(f"\nPrimeras 10 filas:")
    print(df.head(10).to_string(index=False))
    
    print(f"\n{'='*60}")
    print("¡Proceso completado!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
